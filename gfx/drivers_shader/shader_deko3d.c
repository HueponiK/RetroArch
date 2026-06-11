/*  RetroArch - A frontend for libretro.
 *
 *  RetroArch is free software: you can redistribute it and/or modify it under
 *  the terms of the GNU General Public License as published by the Free
 *  Software Foundation, either version 3 of the License, or (at your option)
 *  any later version.
 *
 *  RetroArch is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 *  more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with RetroArch.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <compat/strl.h>
#include <file/file_path.h>
#include <streams/file_stream.h>
#include <string/stdstring.h>
#include <formats/image.h>

#include "shader_deko3d.h"
#include "../../verbosity.h"

/* UAM symbol-redirect macros — must appear before uam.h so the
 * declarations pick up the __uam_ prefixed symbols from the
 * objcopy-mangled libuam.a. */
#ifdef __cplusplus
extern "C" {
#endif
#define uam_init        __uam_uam_init
#define uam_deinit      __uam_uam_deinit
#define uam_compileDksh __uam_uam_compileDksh
#ifdef __cplusplus
}
#endif
#include <uam.h>

/* ====================================================================== *
 *  Internal types
 * ====================================================================== */

/* Semantic IDs for standard slang UBO members. */
enum dk3d_semantic
{
   DK3D_SEM_PARAM = 0,
   DK3D_SEM_MVP,
   DK3D_SEM_OUTPUT_SIZE,
   DK3D_SEM_SOURCE_SIZE,
   DK3D_SEM_ORIGINAL_SIZE,
   DK3D_SEM_FRAME_COUNT,
   DK3D_SEM_FRAME_DIRECTION,
   DK3D_SEM_FINAL_VIEWPORT_SIZE,
   DK3D_SEM_PASS_SIZE,
   DK3D_SEM_LUT_SIZE
};

typedef struct dk3d_uniform_slot
{
   unsigned           offset;
   unsigned           size;
   enum dk3d_semantic sem;
   int                param_idx;
} dk3d_uniform_slot_t;

#define DK3D_MAX_BLOCK_SLOTS 128
#define DK3D_UBO_MAX_SIZE    0x1000
#define DK3D_PUSH_MAX_SIZE   512
#define DK3D_MAX_TEX_BINDINGS 16
/* Per-frame-in-flight copies of the UBO/push/image-descriptor heaps, so the CPU
 * never overwrites a heap region a still-in-flight frame's GPU is still reading
 * (the cause of the SW-path GPU page fault). Matches DK3D_MAX_FRAMES_IN_FLIGHT;
 * the runtime frame index is always < the active frame count, which is <= this. */
#define DK3D_CHAIN_MAX_FRAMES 8

enum dk3d_tex_type
{
   DK3D_TEX_SOURCE = 0,
   DK3D_TEX_ORIGINAL,
   DK3D_TEX_PASS_OUTPUT,
   DK3D_TEX_FEEDBACK,   /* previous-frame output of a pass (double-buffered) */
   DK3D_TEX_LUT
};

typedef struct dk3d_tex_binding
{
   unsigned          binding;
   enum dk3d_tex_type type;
   int               index;
} dk3d_tex_binding_t;

typedef struct dk3d_chain_pass
{
   DkShader    vsh;
   DkShader    fsh;
   DkMemBlock  shader_mem;

   dk3d_image_t fbo;
   /* Previous-frame copy, allocated only when this pass is the target of a
    * *Feedback sampler. fbo/fbo_prev are swapped once per frame so feedback
    * readers sample last frame's output (true double-buffering) instead of
    * the live render target (read-while-write hazard). */
   dk3d_image_t fbo_prev;
   bool         needs_feedback;
   unsigned     fbo_width;
   unsigned     fbo_height;

   struct gfx_fbo_scale scale;
   unsigned     filter;
   enum gfx_wrap_type wrap;
   bool         mipmap;
   bool         float_fbo;
   unsigned     frame_count_mod;
   char         alias[64];

   dk3d_uniform_slot_t ubo_slots[DK3D_MAX_BLOCK_SLOTS];
   unsigned            num_ubo_slots;
   unsigned            ubo_size;

   dk3d_uniform_slot_t push_slots[DK3D_MAX_BLOCK_SLOTS];
   unsigned            num_push_slots;
   unsigned            push_size;

   dk3d_tex_binding_t  tex_bindings[DK3D_MAX_TEX_BINDINGS];
   unsigned            num_tex_bindings;
} dk3d_chain_pass_t;

typedef struct dk3d_chain_lut
{
   dk3d_image_t image;
   char         id[64];
   unsigned     filter;
   enum gfx_wrap_type wrap;
   bool         mipmap;
} dk3d_chain_lut_t;

struct dk3d_filter_chain
{
   DkDevice    device;
   DkQueue     queue;

   struct video_shader *shader;

   dk3d_chain_pass_t passes[DK3D_CHAIN_MAX_PASSES];
   unsigned          num_passes;

   dk3d_chain_lut_t  luts[DK3D_CHAIN_MAX_LUTS];
   unsigned          num_luts;

   /* Sampler + image descriptor heaps (shared, persistent). */
   DkMemBlock  sampler_desc_mem;
   DkGpuAddr   sampler_desc_gpu;
   DkMemBlock  image_desc_mem;
   DkGpuAddr   image_desc_gpu;
   unsigned    num_samplers;
   unsigned    num_images;

   /* Fullscreen quad VBO: 4 vertices × (vec4 Position + vec2 TexCoord). */
   DkMemBlock  vbo_mem;
   DkGpuAddr   vbo_gpu;

   /* Global UBO (binding 0): MVP + sizes + frame info. */
   DkMemBlock  global_mem;
   DkGpuAddr   global_gpu;
   void       *global_ptr;
   float       mvp[16];

   /* Per-pass parameter UBO (binding 1): shader-specific floats. */
   DkMemBlock  push_mem;
   DkGpuAddr   push_gpu;
   void       *push_ptr;

   /* Input from the core. */
   const DkImage *input_image;
   unsigned       input_width;
   unsigned       input_height;

   uint64_t    frame_count;
   int32_t     frame_direction;

   unsigned    vp_x, vp_y, vp_width, vp_height;
   unsigned    swapchain_width, swapchain_height;

   bool        uam_initialized;
};

/* ====================================================================== *
 *  DKSH loading helper (same as deko3d.c's dk3d_load_shader_blob)
 * ====================================================================== */

typedef struct
{
   uint32_t magic;
   uint32_t header_sz;
   uint32_t control_sz;
   uint32_t code_sz;
   uint32_t programs_off;
   uint32_t num_programs;
} dk3d_sc_dksh_header_t;

#define DK3D_SC_DKSH_MAGIC 0x48534B44u

static bool dk3d_sc_load_shader_blob(const uint8_t *blob, size_t blob_sz,
      DkShader *out, DkMemBlock code_mem, uint32_t *io_off)
{
   const dk3d_sc_dksh_header_t *hdr;
   uint32_t off;
   void *code_cpu;
   DkShaderMaker sm;

   if (blob_sz < sizeof(*hdr))
      return false;
   hdr = (const dk3d_sc_dksh_header_t *)blob;
   if (hdr->magic != DK3D_SC_DKSH_MAGIC)
      return false;

   off = (*io_off + DK_SHADER_CODE_ALIGNMENT - 1)
       & ~(DK_SHADER_CODE_ALIGNMENT - 1);
   if (off + hdr->code_sz > dkMemBlockGetSize(code_mem))
      return false;

   code_cpu = (uint8_t *)dkMemBlockGetCpuAddr(code_mem) + off;
   memcpy(code_cpu, blob + hdr->control_sz, hdr->code_sz);

   dkShaderMakerDefaults(&sm, code_mem, off);
   sm.control = blob;
   dkShaderInitialize(out, &sm);

   *io_off = off + hdr->code_sz;
   return true;
}

/* ====================================================================== *
 *  Slang source transform: make UAM-compatible
 * ====================================================================== */

/* Rewrites slang GLSL source for UAM compilation:
 *   1. layout(push_constant) uniform Push → layout(std140, binding=1) uniform Push
 *   2. layout(std140, set = 0, binding = N) → layout(std140, binding = N)
 *   3. layout(set = 0, binding = N) → layout(binding = N)
 * Returns malloc'd string. Caller frees. */
static char *dk3d_slang_transform(const char *src)
{
   char *result;
   size_t src_len, cap, out_len;
   const char *p;

   if (!src)
      return NULL;

   src_len = strlen(src);
   cap     = src_len + 512;
   result  = (char *)malloc(cap);
   if (!result)
      return NULL;

   out_len = 0;
   p       = src;

   while (*p)
   {
      /* Ensure capacity for worst-case expansion. */
      if (out_len + 256 > cap)
      {
         cap *= 2;
         result = (char *)realloc(result, cap);
         if (!result)
            return NULL;
      }

      if (strncmp(p, "layout(", 7) == 0)
      {
         const char *close = strchr(p, ')');
         if (close)
         {
            size_t span = (size_t)(close - p) + 1;
            const char *pc = strstr(p, "push_constant");
            if (pc && pc < close)
            {
               const char *repl = "layout(std140, binding = 1)";
               size_t repl_len = strlen(repl);
               memcpy(result + out_len, repl, repl_len);
               out_len += repl_len;
               p += span;
               continue;
            }
         }
      }

      /* "set = 0, " — strip from any layout() */
      if (strncmp(p, "set = 0, ", 9) == 0)
      {
         p += 9;
         continue;
      }
      /* Alternate spacing: "set=0, " */
      if (strncmp(p, "set=0, ", 7) == 0)
      {
         p += 7;
         continue;
      }

      /* "packed" is a GLSL reserved word but widely used as an identifier
       * in slang shaders. Rename to "packed_v" at word boundaries. */
      if (strncmp(p, "packed", 6) == 0
            && (p == src || (!isalnum((unsigned char)p[-1]) && p[-1] != '_'))
            && !isalnum((unsigned char)p[6])
            && p[6] != '_')
      {
         memcpy(result + out_len, "packed_v", 8);
         out_len += 8;
         p += 6;
         continue;
      }

      result[out_len++] = *p++;
   }

   result[out_len] = '\0';
   return result;
}

/* std140 size/alignment for GLSL types used in slang shaders. */
static void dk3d_glsl_type_info(const char *type,
      unsigned *out_size, unsigned *out_align)
{
   if (string_is_equal(type, "mat4"))
   { *out_size = 64; *out_align = 16; return; }
   if (string_is_equal(type, "vec4"))
   { *out_size = 16; *out_align = 16; return; }
   if (string_is_equal(type, "vec3"))
   { *out_size = 12; *out_align = 16; return; }
   if (string_is_equal(type, "vec2"))
   { *out_size =  8; *out_align =  8; return; }
   *out_size = 4; *out_align = 4;
}

static enum dk3d_semantic dk3d_name_to_semantic(const char *name)
{
   if (string_is_equal(name, "MVP"))            return DK3D_SEM_MVP;
   if (string_is_equal(name, "OutputSize"))     return DK3D_SEM_OUTPUT_SIZE;
   if (string_is_equal(name, "SourceSize"))     return DK3D_SEM_SOURCE_SIZE;
   if (string_is_equal(name, "OriginalSize"))   return DK3D_SEM_ORIGINAL_SIZE;
   if (string_is_equal(name, "FrameCount"))     return DK3D_SEM_FRAME_COUNT;
   if (string_is_equal(name, "FrameDirection")) return DK3D_SEM_FRAME_DIRECTION;
   if (string_is_equal(name, "FinalViewportSize")) return DK3D_SEM_FINAL_VIEWPORT_SIZE;
   return DK3D_SEM_PARAM;
}

/* Parse a uniform block ("uniform UBO" or "uniform Push") from
 * transformed source. Handles comma-separated declarations
 * ("float A, B, C;"). Computes std140 offsets. Maps parameter
 * names to shader->parameters[] by name.
 * Returns number of slots filled, writes total block size to *out_size. */
static unsigned dk3d_parse_uniform_block(const char *src,
      const char *block_name,
      dk3d_uniform_slot_t *slots, unsigned max_slots,
      unsigned *out_size,
      struct video_shader *shader,
      const dk3d_filter_chain_t *chain)
{
   char search[128];
   const char *block, *brace, *end, *p;
   unsigned offset = 0, count = 0;

   snprintf(search, sizeof(search), "uniform %s", block_name);
   block = strstr(src, search);
   if (!block)
   {
      *out_size = 0;
      return 0;
   }

   brace = strchr(block, '{');
   if (!brace) { *out_size = 0; return 0; }
   end = strchr(brace, '}');
   if (!end)   { *out_size = 0; return 0; }

   p = brace + 1;
   while (p < end && count < max_slots)
   {
      char type[32];
      const char *type_start;
      unsigned tsize, talign;
      size_t tlen;

      while (p < end && (*p == ' ' || *p == '\t'
                      || *p == '\n' || *p == '\r'))
         p++;
      if (p >= end)
         break;

      type_start = p;
      while (p < end && *p != ' ' && *p != '\t'
            && *p != '\n' && *p != ',' && *p != ';')
         p++;

      tlen = (size_t)(p - type_start);
      if (tlen >= sizeof(type))
         tlen = sizeof(type) - 1;
      memcpy(type, type_start, tlen);
      type[tlen] = '\0';

      dk3d_glsl_type_info(type, &tsize, &talign);

      while (p < end && count < max_slots)
      {
         char name[64];
         const char *name_start, *name_end;
         size_t nlen;

         while (p < end && (*p == ' ' || *p == '\t'
                         || *p == '\n' || *p == '\r' || *p == ','))
            p++;
         if (p >= end || *p == ';')
            break;

         name_start = p;
         while (p < end && *p != ',' && *p != ';'
               && *p != ' ' && *p != '\t'
               && *p != '\n' && *p != '\r')
            p++;
         name_end = p;

         nlen = (size_t)(name_end - name_start);
         if (nlen == 0)
            break;
         if (nlen >= sizeof(name))
            nlen = sizeof(name) - 1;
         memcpy(name, name_start, nlen);
         name[nlen] = '\0';

         offset = (offset + talign - 1) & ~(talign - 1);

         slots[count].offset    = offset;
         slots[count].size      = tsize;
         slots[count].sem       = dk3d_name_to_semantic(name);
         slots[count].param_idx = -1;

         if (slots[count].sem == DK3D_SEM_PARAM && shader)
         {
            unsigned j;
            for (j = 0; j < shader->num_parameters; j++)
            {
               if (string_is_equal(name, shader->parameters[j].id))
               {
                  slots[count].param_idx = (int)j;
                  break;
               }
            }
         }

         if (slots[count].sem == DK3D_SEM_PARAM
               && slots[count].param_idx == -1 && chain)
         {
            size_t namelen = strlen(name);
            if (namelen > 4
                  && string_is_equal(name + namelen - 4, "Size"))
            {
               char base[64];
               unsigned j;
               strlcpy(base, name, sizeof(base));
               base[namelen - 4] = '\0';

               for (j = 0; j < chain->num_passes; j++)
               {
                  if (chain->passes[j].alias[0]
                        && string_is_equal(base, chain->passes[j].alias))
                  {
                     slots[count].sem       = DK3D_SEM_PASS_SIZE;
                     slots[count].param_idx = (int)j;
                     break;
                  }
               }

               if (slots[count].sem == DK3D_SEM_PARAM)
               {
                  for (j = 0; j < chain->num_luts; j++)
                  {
                     if (string_is_equal(base, chain->luts[j].id))
                     {
                        slots[count].sem       = DK3D_SEM_LUT_SIZE;
                        slots[count].param_idx = (int)j;
                        break;
                     }
                  }
               }
            }
         }

         offset += tsize;
         count++;
      }

      if (p < end && *p == ';')
         p++;
   }

   *out_size = (offset + 15u) & ~15u;
   return count;
}

/* Parse both UBO (binding 0) and Push (binding 1) blocks
 * from transformed shader source into per-pass slot arrays. */
static void dk3d_parse_pass_uniforms(const char *src,
      dk3d_chain_pass_t *pass, struct video_shader *shader,
      const dk3d_filter_chain_t *chain)
{
   pass->num_ubo_slots = dk3d_parse_uniform_block(src, "UBO",
         pass->ubo_slots, DK3D_MAX_BLOCK_SLOTS,
         &pass->ubo_size, shader, chain);

   pass->num_push_slots = dk3d_parse_uniform_block(src, "Push",
         pass->push_slots, DK3D_MAX_BLOCK_SLOTS,
         &pass->push_size, shader, chain);
}

static void dk3d_parse_pass_textures(const char *src,
      dk3d_chain_pass_t *pass,
      dk3d_filter_chain_t *chain, unsigned this_pass)
{
   const char *p = src;
   pass->num_tex_bindings = 0;

   while (pass->num_tex_bindings < DK3D_MAX_TEX_BINDINGS)
   {
      const char *decl, *bp, *np, *name_end;
      char name[64];
      unsigned binding;
      size_t nlen;
      dk3d_tex_binding_t *tb;

      decl = strstr(p, "sampler2D");
      if (!decl)
         break;

      /* Walk backwards to find binding = N in the layout qualifier. */
      {
         const char *layout = decl;
         bool found = false;
         while (layout > src && (layout - src) > 0)
         {
            layout--;
            if (*layout == '\n' || *layout == ';')
               break;
            if (!strncmp(layout, "binding", 7))
            {
               bp = layout + 7;
               while (*bp == ' ' || *bp == '=')
                  bp++;
               binding = (unsigned)strtoul(bp, NULL, 10);
               found = true;
               break;
            }
         }
         if (!found)
         {
            p = decl + 9;
            continue;
         }
      }

      /* Extract the sampler name. */
      np = decl + 9; /* skip "sampler2D" */
      while (*np == ' ' || *np == '\t')
         np++;
      name_end = np;
      while (*name_end && *name_end != ';' && *name_end != ' '
            && *name_end != '\t' && *name_end != '\n'
            && *name_end != '[')
         name_end++;
      nlen = (size_t)(name_end - np);
      if (nlen == 0 || nlen >= sizeof(name))
      {
         p = decl + 9;
         continue;
      }
      memcpy(name, np, nlen);
      name[nlen] = '\0';

      tb = &pass->tex_bindings[pass->num_tex_bindings];
      tb->binding = binding;
      tb->index   = -1;

      if (string_is_equal(name, "Source"))
         tb->type = DK3D_TEX_SOURCE;
      else if (string_is_equal(name, "Original")
            || string_is_equal(name, "OriginalHistory0"))
         tb->type = DK3D_TEX_ORIGINAL;
      else
      {
         unsigned j;
         bool resolved = false;

         /* Check pass aliases. */
         for (j = 0; j < this_pass; j++)
         {
            if (*chain->passes[j].alias
                  && string_is_equal(name, chain->passes[j].alias))
            {
               tb->type  = DK3D_TEX_PASS_OUTPUT;
               tb->index = (int)j;
               resolved  = true;
               break;
            }
         }

         /* Check PassOutputN naming. */
         if (!resolved && !strncmp(name, "PassOutput", 10))
         {
            tb->type  = DK3D_TEX_PASS_OUTPUT;
            tb->index = (int)strtoul(name + 10, NULL, 10);
            resolved  = true;
         }

         /* *Feedback: previous-frame output of the named pass. Resolves to a
          * dedicated history surface (passes[j].fbo_prev) which is swapped
          * with passes[j].fbo once per frame, so the read samples last
          * frame's output rather than the live render target. Flags the
          * target pass so its fbo_prev gets allocated. The named pass may be
          * this pass itself (self-feedback, e.g. perf-pass) or any other. */
         if (!resolved)
         {
            size_t name_len = strlen(name);
            if (name_len > 8 && !strcmp(name + name_len - 8, "Feedback"))
            {
               char base[64];
               memcpy(base, name, name_len - 8);
               base[name_len - 8] = '\0';
               for (j = 0; j < chain->num_passes; j++)
               {
                  if (*chain->passes[j].alias
                        && string_is_equal(base, chain->passes[j].alias))
                  {
                     tb->type  = DK3D_TEX_FEEDBACK;
                     tb->index = (int)j;
                     chain->passes[j].needs_feedback = true;
                     resolved  = true;
                     break;
                  }
               }
            }
         }

         /* Check LUT names. */
         if (!resolved)
         {
            for (j = 0; j < chain->num_luts; j++)
            {
               if (string_is_equal(name, chain->luts[j].id))
               {
                  tb->type  = DK3D_TEX_LUT;
                  tb->index = (int)j;
                  resolved  = true;
                  break;
               }
            }
         }

         if (!resolved)
         {
            RARCH_WARN("[deko3d shader] Unresolved texture: %s (binding %u)\n",
                  name, binding);
            p = decl + 9;
            continue;
         }
      }

      pass->num_tex_bindings++;
      p = decl + 9;
   }
}

/* Split a slang source at #pragma stage directives.
 * Returns malloc'd vertex and fragment strings. */
static bool dk3d_slang_split_stages(const char *source,
      char **out_vert, char **out_frag)
{
   const char *vert_start = NULL;
   const char *frag_start = NULL;
   const char *p;
   size_t vert_len, frag_len;
   char *header = NULL;
   size_t header_len = 0;

   /* Find #pragma stage vertex */
   p = strstr(source, "#pragma stage vertex");
   if (!p)
      return false;

   /* Everything before the first #pragma stage is shared header. */
   header_len = (size_t)(p - source);
   header = (char *)malloc(header_len + 1);
   memcpy(header, source, header_len);
   header[header_len] = '\0';

   /* Skip past the directive line */
   vert_start = strchr(p, '\n');
   if (vert_start)
      vert_start++;
   else
      vert_start = p + strlen("#pragma stage vertex");

   /* Find #pragma stage fragment */
   p = strstr(vert_start, "#pragma stage fragment");
   if (!p)
   {
      free(header);
      return false;
   }

   vert_len = (size_t)(p - vert_start);
   frag_start = strchr(p, '\n');
   if (frag_start)
      frag_start++;
   else
      frag_start = p + strlen("#pragma stage fragment");

   frag_len = strlen(frag_start);

   /* Assemble: header + stage body */
   *out_vert = (char *)malloc(header_len + vert_len + 1);
   memcpy(*out_vert, header, header_len);
   memcpy(*out_vert + header_len, vert_start, vert_len);
   (*out_vert)[header_len + vert_len] = '\0';

   *out_frag = (char *)malloc(header_len + frag_len + 1);
   memcpy(*out_frag, header, header_len);
   memcpy(*out_frag + header_len, frag_start, frag_len);
   (*out_frag)[header_len + frag_len] = '\0';

   free(header);
   return true;
}

/* Strip #pragma format, #pragma name, #pragma parameter lines
 * (RetroArch metadata that UAM's GLSL parser rejects).
 * Modifies the string in place. */
static void dk3d_strip_slang_pragmas(char *src)
{
   char *rd = src, *wr = src;
   while (*rd)
   {
      if (rd[0] == '#' && strncmp(rd, "#pragma ", 8) == 0)
      {
         const char *after = rd + 8;
         while (*after == ' ')
            after++;
         if (strncmp(after, "format", 6) == 0
               || strncmp(after, "name", 4) == 0
               || strncmp(after, "parameter", 9) == 0)
         {
            while (*rd && *rd != '\n')
               rd++;
            if (*rd == '\n')
               rd++;
            continue;
         }
      }
      *wr++ = *rd++;
   }
   *wr = '\0';
}

/* Resolve #include "file" directives relative to base_dir.
 * Returns a malloc'd string with all includes inlined.
 * Recursive up to 16 levels deep. */
static char *dk3d_resolve_includes(const char *src, const char *base_dir,
      int depth)
{
   const char *p = src;
   char *result = NULL;
   size_t cap = 0, len = 0;

   if (depth > 16 || !src)
      return strdup(src ? src : "");

   cap = strlen(src) + 1024;
   result = (char *)malloc(cap);
   if (!result)
      return NULL;
   len = 0;

   while (*p)
   {
      if (p[0] == '#')
      {
         const char *q = p + 1;
         while (*q == ' ')
            q++;
         if (strncmp(q, "include", 7) == 0)
         {
            q += 7;
            while (*q == ' ')
               q++;
            if (*q == '"')
            {
               const char *name_start = ++q;
               while (*q && *q != '"' && *q != '\n')
                  q++;
               if (*q == '"')
               {
                  char inc_name[PATH_MAX_LENGTH];
                  char inc_path[PATH_MAX_LENGTH];
                  char inc_dir[PATH_MAX_LENGTH];
                  int64_t inc_len = 0;
                  char *inc_raw = NULL;
                  char *inc_resolved = NULL;
                  size_t name_len = (size_t)(q - name_start);

                  if (name_len >= sizeof(inc_name))
                     name_len = sizeof(inc_name) - 1;
                  memcpy(inc_name, name_start, name_len);
                  inc_name[name_len] = '\0';

                  fill_pathname_join_special(inc_path, base_dir,
                        inc_name, sizeof(inc_path));

                  if (filestream_read_file(inc_path,
                           (void **)&inc_raw, &inc_len) > 0 && inc_len > 0)
                  {
                     fill_pathname_basedir(inc_dir, inc_path,
                           sizeof(inc_dir));
                     inc_resolved = dk3d_resolve_includes(
                           inc_raw, inc_dir, depth + 1);
                     free(inc_raw);

                     if (inc_resolved)
                     {
                        size_t rl = strlen(inc_resolved);
                        while (len + rl + 2 > cap)
                        {
                           cap *= 2;
                           result = (char *)realloc(result, cap);
                        }
                        memcpy(result + len, inc_resolved, rl);
                        len += rl;
                        if (rl > 0 && inc_resolved[rl - 1] != '\n')
                           result[len++] = '\n';
                        free(inc_resolved);
                     }
                  }
                  else
                  {
                     RARCH_WARN("[deko3d shader] #include not found: %s\n",
                           inc_path);
                  }

                  q++;
                  while (*q && *q != '\n')
                     q++;
                  if (*q == '\n')
                     q++;
                  p = q;
                  continue;
               }
            }
         }
      }

      if (len + 2 > cap)
      {
         cap *= 2;
         result = (char *)realloc(result, cap);
      }
      result[len++] = *p++;
   }
   result[len] = '\0';
   return result;
}

/* ====================================================================== *
 *  Load two DKSH blobs (vsh + fsh) into a DkShader pair
 * ====================================================================== */

static bool dk3d_load_shader_pair(
      const uint8_t *vsh_blob, uint32_t vsh_size,
      const uint8_t *fsh_blob, uint32_t fsh_size,
      DkShader *vsh, DkShader *fsh, DkMemBlock *shader_mem,
      DkDevice device)
{
   DkMemBlockMaker mm;
   uint32_t total_size, code_off = 0;

   total_size = ((vsh_size + DK_SHADER_CODE_ALIGNMENT - 1)
                  & ~(DK_SHADER_CODE_ALIGNMENT - 1))
              + ((fsh_size + DK_SHADER_CODE_ALIGNMENT - 1)
                  & ~(DK_SHADER_CODE_ALIGNMENT - 1));
   total_size = (total_size + 0xFFFu) & ~0xFFFu;

   dkMemBlockMakerDefaults(&mm, device, total_size);
   mm.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached
            | DkMemBlockFlags_Code;
   *shader_mem = dkMemBlockCreate(&mm);
   if (!*shader_mem)
      return false;

   if (!dk3d_sc_load_shader_blob(vsh_blob, vsh_size, vsh, *shader_mem, &code_off))
      goto fail;
   if (!dk3d_sc_load_shader_blob(fsh_blob, fsh_size, fsh, *shader_mem, &code_off))
      goto fail;

   dkMemBlockFlushCpuCache(*shader_mem, 0, total_size);
   return true;

fail:
   dkMemBlockDestroy(*shader_mem);
   *shader_mem = NULL;
   return false;
}

/* ====================================================================== *
 *  Precompiled DKSH cache: <path>.vert.dksh / <path>.frag.dksh
 * ====================================================================== */

static bool dk3d_load_precompiled(
      const char *slang_path,
      DkShader *vsh, DkShader *fsh, DkMemBlock *shader_mem,
      DkDevice device)
{
   char vert_path[PATH_MAX_LENGTH];
   char frag_path[PATH_MAX_LENGTH];
   uint8_t *vsh_blob = NULL, *fsh_blob = NULL;
   int64_t vsh_len = 0, fsh_len = 0;
   bool ok;

   snprintf(vert_path, sizeof(vert_path), "%s.vert.dksh", slang_path);
   snprintf(frag_path, sizeof(frag_path), "%s.frag.dksh", slang_path);

   if (filestream_read_file(vert_path, (void **)&vsh_blob, &vsh_len) <= 0
         || vsh_len <= 0)
      return false;

   if (filestream_read_file(frag_path, (void **)&fsh_blob, &fsh_len) <= 0
         || fsh_len <= 0)
   {
      free(vsh_blob);
      return false;
   }

   ok = dk3d_load_shader_pair(vsh_blob, (uint32_t)vsh_len,
         fsh_blob, (uint32_t)fsh_len,
         vsh, fsh, shader_mem, device);

   free(vsh_blob);
   free(fsh_blob);
   return ok;
}

/* ====================================================================== *
 *  Compile a single slang shader file into vsh + fsh DkShader pair.
 *  Tries precompiled .dksh cache first, falls back to runtime UAM.
 * ====================================================================== */

static bool dk3d_compile_slang_pass(
      const char *path,
      DkShader *vsh, DkShader *fsh, DkMemBlock *shader_mem,
      DkDevice device, bool uam_available)
{
   int64_t len = 0;
   char *raw_source = NULL;
   char *transformed = NULL;
   char *vert_src = NULL, *frag_src = NULL;
   uint8_t *vsh_dksh = NULL, *fsh_dksh = NULL;
   uint32_t vsh_size = 0, fsh_size = 0;
   bool success = false;

   /* Fast path: precompiled DKSH blobs alongside the .slang file. */
   if (dk3d_load_precompiled(path, vsh, fsh, shader_mem, device))
   {
      RARCH_LOG("[deko3d shader] Loaded precompiled: %s\n", path);
      return true;
   }

   if (!uam_available)
   {
      RARCH_ERR("[deko3d shader] No precompiled cache and UAM not available: %s\n", path);
      return false;
   }

   /* Runtime compilation via libuam. */
   {
      char base_dir[PATH_MAX_LENGTH];
      char *resolved = NULL;

      if (filestream_read_file(path, (void **)&raw_source, &len) <= 0 || len <= 0)
      {
         RARCH_ERR("[deko3d shader] Failed to read: %s\n", path);
         return false;
      }

      fill_pathname_basedir(base_dir, path, sizeof(base_dir));
      resolved = dk3d_resolve_includes(raw_source, base_dir, 0);
      free(raw_source);
      if (!resolved)
         return false;

      dk3d_strip_slang_pragmas(resolved);
      transformed = dk3d_slang_transform(resolved);
      free(resolved);
   }
   if (!transformed)
      return false;

   if (!dk3d_slang_split_stages(transformed, &vert_src, &frag_src))
   {
      RARCH_ERR("[deko3d shader] Failed to split stages: %s\n", path);
      free(transformed);
      return false;
   }
   free(transformed);

   if (!uam_compileDksh(uam_pipeline_stage_vertex, vert_src, 2,
            &vsh_dksh, &vsh_size))
   {
      char dump[PATH_MAX_LENGTH];
      RARCH_ERR("[deko3d shader] Vertex compile failed: %s\n", path);
      snprintf(dump, sizeof(dump), "%s.fail.vert", path);
      filestream_write_file(dump, vert_src, strlen(vert_src));
      goto cleanup;
   }

   if (!uam_compileDksh(uam_pipeline_stage_fragment, frag_src, 2,
            &fsh_dksh, &fsh_size))
   {
      char dump[PATH_MAX_LENGTH];
      RARCH_ERR("[deko3d shader] Fragment compile failed: %s\n", path);
      snprintf(dump, sizeof(dump), "%s.fail.frag", path);
      filestream_write_file(dump, frag_src, strlen(frag_src));
      goto cleanup;
   }

   success = dk3d_load_shader_pair(vsh_dksh, vsh_size,
         fsh_dksh, fsh_size,
         vsh, fsh, shader_mem, device);

   if (success)
   {
      char cache_path[PATH_MAX_LENGTH];

      snprintf(cache_path, sizeof(cache_path), "%s.vert.dksh", path);
      if (filestream_write_file(cache_path, vsh_dksh, vsh_size))
         RARCH_LOG("[deko3d shader] Cached: %s\n", cache_path);

      snprintf(cache_path, sizeof(cache_path), "%s.frag.dksh", path);
      if (filestream_write_file(cache_path, fsh_dksh, fsh_size))
         RARCH_LOG("[deko3d shader] Cached: %s\n", cache_path);

      RARCH_LOG("[deko3d shader] Runtime compiled: %s\n", path);
   }

cleanup:
   free(vert_src);
   free(frag_src);
   free(vsh_dksh);
   free(fsh_dksh);
   return success;
}

/* ====================================================================== *
 *  LUT texture loading (PNG → dk3d_image_t)
 * ====================================================================== */

static bool dk3d_load_lut_image(DkDevice device, DkQueue queue,
      const char *path, dk3d_chain_lut_t *lut)
{
   struct texture_image img = {0};
   DkMemBlockMaker mm;
   DkMemBlock staging = NULL;
   uint32_t staging_size;
   DkCmdBuf cmd = NULL;
   DkMemBlock cmd_mem = NULL;

   img.supports_rgba = true;
   if (!image_texture_load(&img, path))
   {
      RARCH_ERR("[deko3d shader] Failed to load LUT: %s\n", path);
      return false;
   }

   if (!dk3d_create_image_2d(device, img.width, img.height,
            DkImageFormat_RGBA8_Unorm,
            DkImageFlags_UsageRender | DkImageFlags_Usage2DEngine,
            &lut->image))
   {
      image_texture_free(&img);
      return false;
   }

   staging_size = img.width * img.height * 4;
   staging_size = (staging_size + 0xFFFu) & ~0xFFFu;

   dkMemBlockMakerDefaults(&mm, device, staging_size);
   mm.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
   staging = dkMemBlockCreate(&mm);
   if (!staging)
   {
      dk3d_destroy_image(&lut->image);
      image_texture_free(&img);
      return false;
   }

   memcpy(dkMemBlockGetCpuAddr(staging), img.pixels,
         img.width * img.height * 4);

   /* One-shot command buffer for the upload. */
   dkMemBlockMakerDefaults(&mm, device, 0x1000);
   mm.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
   cmd_mem = dkMemBlockCreate(&mm);
   if (!cmd_mem)
   {
      dkMemBlockDestroy(staging);
      dk3d_destroy_image(&lut->image);
      image_texture_free(&img);
      return false;
   }

   {
      DkCmdBufMaker cbm;
      dkCmdBufMakerDefaults(&cbm, device);
      cmd = dkCmdBufCreate(&cbm);
   }
   dkCmdBufAddMemory(cmd, cmd_mem, 0, dkMemBlockGetSize(cmd_mem));

   {
      DkImageView dst_view;
      DkCopyBuf src_buf;
      DkImageRect dst_rect;

      dkImageViewDefaults(&dst_view, &lut->image.image);
      src_buf.addr   = dkMemBlockGetGpuAddr(staging);
      src_buf.rowLength   = 0;
      src_buf.imageHeight = 0;
      dst_rect.x = dst_rect.y = dst_rect.z = 0;
      dst_rect.width  = img.width;
      dst_rect.height = img.height;
      dst_rect.depth  = 1;

      dkCmdBufCopyBufferToImage(cmd, &src_buf, &dst_view, &dst_rect, 0);
   }

   {
      DkCmdList list = dkCmdBufFinishList(cmd);
      dkQueueSubmitCommands(queue, list);
      dkQueueWaitIdle(queue);
   }

   dkCmdBufDestroy(cmd);
   dkMemBlockDestroy(cmd_mem);
   dkMemBlockDestroy(staging);
   image_texture_free(&img);

   return true;
}

/* ====================================================================== *
 *  FBO management
 * ====================================================================== */

static void dk3d_compute_pass_fbo_size(
      const struct gfx_fbo_scale *scale,
      unsigned input_w, unsigned input_h,
      unsigned vp_w, unsigned vp_h,
      unsigned *out_w, unsigned *out_h)
{
   switch (scale->type_x)
   {
      case RARCH_SCALE_INPUT:
         *out_w = (unsigned)(input_w * scale->scale_x);
         break;
      case RARCH_SCALE_ABSOLUTE:
         *out_w = scale->abs_x;
         break;
      case RARCH_SCALE_VIEWPORT:
         *out_w = (unsigned)(vp_w * scale->scale_x);
         break;
   }

   switch (scale->type_y)
   {
      case RARCH_SCALE_INPUT:
         *out_h = (unsigned)(input_h * scale->scale_y);
         break;
      case RARCH_SCALE_ABSOLUTE:
         *out_h = scale->abs_y;
         break;
      case RARCH_SCALE_VIEWPORT:
         *out_h = (unsigned)(vp_h * scale->scale_y);
         break;
   }

   if (*out_w < 1) *out_w = 1;
   if (*out_h < 1) *out_h = 1;
}

static bool dk3d_chain_create_pass_fbo(DkDevice device, DkQueue queue,
      dk3d_chain_pass_t *pass, unsigned w, unsigned h)
{
   DkImageFormat fmt   = pass->float_fbo ? DkImageFormat_RGBA16_Float
                                         : DkImageFormat_RGBA8_Unorm;
   uint32_t      flags = DkImageFlags_UsageRender | DkImageFlags_Usage2DEngine
                       | DkImageFlags_HwCompression;

   if (pass->fbo.memblock
         && (pass->fbo_width != w || pass->fbo_height != h))
   {
      /* The old FBO can still be referenced by in-flight GPU work when a
       * resolution change frees it here (a core reset walks the BIOS
       * resolution ladder). Drain the queue first to avoid a use-after-free
       * that faults at the next acquire — mirrors the teardown path. Resize
       * is rare, so the stall is harmless. */
      dkQueueWaitIdle(queue);
      dk3d_destroy_image(&pass->fbo);
   }

   if (!pass->fbo.memblock)
   {
      if (!dk3d_create_image_2d(device, w, h, fmt, flags, &pass->fbo))
         return false;
      pass->fbo_width  = w;
      pass->fbo_height = h;
   }

   /* Match the feedback history buffer to the render target's size. Both are
    * kept identical so the once-per-frame fbo<->fbo_prev swap is valid. */
   if (pass->needs_feedback)
   {
      if (pass->fbo_prev.memblock
            && (pass->fbo_prev.width != w || pass->fbo_prev.height != h))
      {
         dkQueueWaitIdle(queue);
         dk3d_destroy_image(&pass->fbo_prev);
      }
      if (!pass->fbo_prev.memblock)
      {
         if (!dk3d_create_image_2d(device, w, h, fmt, flags, &pass->fbo_prev))
            return false;
      }
   }

   return true;
}

/* ====================================================================== *
 *  Descriptor heap helpers
 * ====================================================================== */

static void dk3d_chain_rebuild_descs(dk3d_filter_chain_t *chain)
{
   unsigned i;
   DkSamplerDescriptor *samp;

   samp = (DkSamplerDescriptor *)dkMemBlockGetCpuAddr(chain->sampler_desc_mem);

   chain->num_samplers = 0;
   chain->num_images   = 0;

   /* Slot 0: default sampler (linear, clamp). Used as a fallback. */
   {
      DkSampler s;
      dkSamplerDefaults(&s);
      s.minFilter = DkFilter_Linear;
      s.magFilter = DkFilter_Linear;
      s.wrapMode[0] = DkWrapMode_ClampToEdge;
      s.wrapMode[1] = DkWrapMode_ClampToEdge;
      dkSamplerDescriptorInitialize(&samp[chain->num_samplers++], &s);
   }
   /* Slot 1: nearest, clamp. */
   {
      DkSampler s;
      dkSamplerDefaults(&s);
      s.minFilter = DkFilter_Nearest;
      s.magFilter = DkFilter_Nearest;
      s.wrapMode[0] = DkWrapMode_ClampToEdge;
      s.wrapMode[1] = DkWrapMode_ClampToEdge;
      dkSamplerDescriptorInitialize(&samp[chain->num_samplers++], &s);
   }

   /* Per-pass samplers. */
   for (i = 0; i < chain->num_passes; i++)
   {
      dk3d_chain_pass_t *p = &chain->passes[i];
      DkSampler s;
      dkSamplerDefaults(&s);
      s.minFilter = (p->filter == RARCH_FILTER_LINEAR)
                  ? DkFilter_Linear : DkFilter_Nearest;
      s.magFilter = s.minFilter;

      switch (p->wrap)
      {
         case RARCH_WRAP_REPEAT:
            s.wrapMode[0] = s.wrapMode[1] = DkWrapMode_Repeat;
            break;
         case RARCH_WRAP_MIRRORED_REPEAT:
            s.wrapMode[0] = s.wrapMode[1] = DkWrapMode_MirroredRepeat;
            break;
         case RARCH_WRAP_EDGE:
            s.wrapMode[0] = s.wrapMode[1] = DkWrapMode_ClampToEdge;
            break;
         default:
            s.wrapMode[0] = s.wrapMode[1] = DkWrapMode_ClampToBorder;
            break;
      }
      dkSamplerDescriptorInitialize(&samp[chain->num_samplers++], &s);
   }

   /* Per-LUT samplers. */
   for (i = 0; i < chain->num_luts; i++)
   {
      dk3d_chain_lut_t *l = &chain->luts[i];
      DkSampler s;
      dkSamplerDefaults(&s);
      s.minFilter = (l->filter == RARCH_FILTER_LINEAR)
                  ? DkFilter_Linear : DkFilter_Nearest;
      s.magFilter = s.minFilter;
      s.wrapMode[0] = s.wrapMode[1] = DkWrapMode_ClampToEdge;
      dkSamplerDescriptorInitialize(&samp[chain->num_samplers++], &s);
   }
}

/* ====================================================================== *
 *  Public API: creation
 * ====================================================================== */

static dk3d_filter_chain_t *dk3d_chain_alloc(
      const struct dk3d_filter_chain_create_info *info)
{
   dk3d_filter_chain_t *chain;
   DkMemBlockMaker mm;

   chain = (dk3d_filter_chain_t *)calloc(1, sizeof(*chain));
   if (!chain)
      return NULL;

   chain->device           = info->device;
   chain->queue            = info->queue;
   chain->swapchain_width  = info->swapchain_width;
   chain->swapchain_height = info->swapchain_height;
   chain->frame_direction  = 1;

   /* Identity MVP. */
   memset(chain->mvp, 0, sizeof(chain->mvp));
   chain->mvp[0] = chain->mvp[5] = chain->mvp[10] = chain->mvp[15] = 1.0f;

   /* Global UBO pool (binding 0): one slot per pass so each pass keeps
    * its own OutputSize / SourceSize / etc. until the GPU executes. */
   {
      uint32_t global_pool = DK3D_UBO_MAX_SIZE * DK3D_CHAIN_MAX_PASSES
                           * DK3D_CHAIN_MAX_FRAMES;
      global_pool = (global_pool + 0xFFFu) & ~0xFFFu;
      dkMemBlockMakerDefaults(&mm, info->device, global_pool);
      mm.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
      chain->global_mem = dkMemBlockCreate(&mm);
      if (!chain->global_mem)
         goto fail;
      chain->global_gpu = dkMemBlockGetGpuAddr(chain->global_mem);
      chain->global_ptr = dkMemBlockGetCpuAddr(chain->global_mem);
      memset(chain->global_ptr, 0, global_pool);
   }

   /* Push UBO pool (binding 1): one slot per pass. */
   {
      uint32_t push_size = DK3D_PUSH_MAX_SIZE * DK3D_CHAIN_MAX_PASSES
                         * DK3D_CHAIN_MAX_FRAMES;
      push_size = (push_size + 0xFFFu) & ~0xFFFu;
      dkMemBlockMakerDefaults(&mm, info->device, push_size);
      mm.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
      chain->push_mem = dkMemBlockCreate(&mm);
      if (!chain->push_mem)
         goto fail;
      chain->push_gpu = dkMemBlockGetGpuAddr(chain->push_mem);
      chain->push_ptr = dkMemBlockGetCpuAddr(chain->push_mem);
   }

   /* Sampler descriptor heap. */
   {
      uint32_t samp_size = sizeof(DkSamplerDescriptor) * DK3D_CHAIN_MAX_SAMPLERS;
      samp_size = (samp_size + 0xFFFu) & ~0xFFFu;
      dkMemBlockMakerDefaults(&mm, info->device, samp_size);
      mm.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
      chain->sampler_desc_mem = dkMemBlockCreate(&mm);
      if (!chain->sampler_desc_mem)
         goto fail;
      chain->sampler_desc_gpu = dkMemBlockGetGpuAddr(chain->sampler_desc_mem);
   }

   /* Image descriptor heap. */
   {
      uint32_t img_size = sizeof(DkImageDescriptor)
                        * (DK3D_CHAIN_MAX_PASSES * DK3D_MAX_TEX_BINDINGS)
                        * DK3D_CHAIN_MAX_FRAMES;
      img_size = (img_size + 0xFFFu) & ~0xFFFu;
      dkMemBlockMakerDefaults(&mm, info->device, img_size);
      mm.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
      chain->image_desc_mem = dkMemBlockCreate(&mm);
      if (!chain->image_desc_mem)
         goto fail;
      chain->image_desc_gpu = dkMemBlockGetGpuAddr(chain->image_desc_mem);
   }

   /* Fullscreen quad VBO: triangle strip, stride 24 (4f pos + 2f uv).
    * Slang vertex shaders expect:
    *   layout(location = 0) in vec4 Position;
    *   layout(location = 1) in vec2 TexCoord;
    *
    * Two quads are stored back-to-back:
    *   verts 0..3 (offset 0)  : final/to-screen quad, uv.y inverted vs clip.y.
    *                            Cancels the scanout (present) vertical flip so
    *                            the displayed image is upright.
    *   verts 4..7 (offset 96) : offscreen quad, uv.y follows clip.y directly so
    *                            an FBO stores its source orientation-NEUTRAL
    *                            (row 0 = source row 0). Keeping offscreen passes
    *                            neutral makes the chain independent of pass count
    *                            and keeps every FBO/PassOutput/Original in one
    *                            shared orientation (multi-input passes line up). */
   {
      float *v;
      dkMemBlockMakerDefaults(&mm, info->device, 0x1000);
      mm.flags = DkMemBlockFlags_CpuUncached | DkMemBlockFlags_GpuCached;
      chain->vbo_mem = dkMemBlockCreate(&mm);
      if (!chain->vbo_mem)
         goto fail;
      chain->vbo_gpu = dkMemBlockGetGpuAddr(chain->vbo_mem);
      v = (float *)dkMemBlockGetCpuAddr(chain->vbo_mem);
      /* --- final quad (uv.y inverted) --- */
      v[ 0] = -1.0f; v[ 1] = -1.0f; v[ 2] = 0.0f; v[ 3] = 1.0f;
      v[ 4] =  0.0f; v[ 5] =  1.0f;
      v[ 6] =  1.0f; v[ 7] = -1.0f; v[ 8] = 0.0f; v[ 9] = 1.0f;
      v[10] =  1.0f; v[11] =  1.0f;
      v[12] = -1.0f; v[13] =  1.0f; v[14] = 0.0f; v[15] = 1.0f;
      v[16] =  0.0f; v[17] =  0.0f;
      v[18] =  1.0f; v[19] =  1.0f; v[20] = 0.0f; v[21] = 1.0f;
      v[22] =  1.0f; v[23] =  0.0f;
      /* --- offscreen quad (uv.y follows clip.y, orientation-neutral) --- */
      v[24] = -1.0f; v[25] = -1.0f; v[26] = 0.0f; v[27] = 1.0f;
      v[28] =  0.0f; v[29] =  0.0f;
      v[30] =  1.0f; v[31] = -1.0f; v[32] = 0.0f; v[33] = 1.0f;
      v[34] =  1.0f; v[35] =  0.0f;
      v[36] = -1.0f; v[37] =  1.0f; v[38] = 0.0f; v[39] = 1.0f;
      v[40] =  0.0f; v[41] =  1.0f;
      v[42] =  1.0f; v[43] =  1.0f; v[44] = 0.0f; v[45] = 1.0f;
      v[46] =  1.0f; v[47] =  1.0f;
   }

   return chain;

fail:
   dk3d_filter_chain_free(chain);
   return NULL;
}

dk3d_filter_chain_t *dk3d_filter_chain_create_default(
      const struct dk3d_filter_chain_create_info *info,
      bool linear)
{
   dk3d_filter_chain_t *chain = dk3d_chain_alloc(info);
   if (!chain)
      return NULL;

   /* No passes = the driver's existing blit pipeline handles everything.
    * The chain exists only so set_shader has something to point at. */
   chain->num_passes = 0;
   dk3d_chain_rebuild_descs(chain);
   return chain;
}

dk3d_filter_chain_t *dk3d_filter_chain_create_from_preset(
      const struct dk3d_filter_chain_create_info *info,
      const char *preset_path)
{
   unsigned i;
   dk3d_filter_chain_t *chain;
   struct video_shader *shader;
   bool is_preset;
   enum rarch_shader_type type;

   type = video_shader_get_type_from_ext(
         path_get_extension(preset_path), &is_preset);

   if (type != RARCH_SHADER_SLANG)
   {
      RARCH_ERR("[deko3d shader] Only .slangp presets are supported, got: %s\n",
            preset_path);
      return NULL;
   }

   chain = dk3d_chain_alloc(info);
   if (!chain)
      return NULL;

   /* Parse the preset. */
   shader = (struct video_shader *)calloc(1, sizeof(*shader));
   if (!shader)
   {
      dk3d_filter_chain_free(chain);
      return NULL;
   }

   if (!video_shader_load_preset_into_shader(preset_path, shader))
   {
      RARCH_ERR("[deko3d shader] Failed to parse preset: %s\n", preset_path);
      free(shader);
      dk3d_filter_chain_free(chain);
      return NULL;
   }

   chain->shader = shader;

   /* Initialize UAM compiler. */
   if (!chain->uam_initialized)
   {
      uam_init();
      chain->uam_initialized = true;
   }

   RARCH_LOG("[deko3d shader] Compiling %u passes from %s\n",
         shader->passes, preset_path);

   /* Pre-populate pass aliases + num_passes so the UBO parser can
    * resolve *Size uniforms (e.g. LinearizePassSize) during compilation. */
   chain->num_passes = (shader->passes < DK3D_CHAIN_MAX_PASSES)
                     ? shader->passes : DK3D_CHAIN_MAX_PASSES;
   for (i = 0; i < chain->num_passes; i++)
   {
      if (*shader->pass[i].alias)
         strlcpy(chain->passes[i].alias, shader->pass[i].alias,
               sizeof(chain->passes[i].alias));
   }

   /* Pre-populate LUT ids so texture binding parser can resolve names. */
   for (i = 0; i < shader->luts && i < DK3D_CHAIN_MAX_LUTS; i++)
      strlcpy(chain->luts[i].id, shader->lut[i].id, sizeof(chain->luts[i].id));
   chain->num_luts = (shader->luts < DK3D_CHAIN_MAX_LUTS)
                   ? shader->luts : DK3D_CHAIN_MAX_LUTS;

   /* Compile each pass. */
   for (i = 0; i < shader->passes && i < DK3D_CHAIN_MAX_PASSES; i++)
   {
      struct video_shader_pass *sp = &shader->pass[i];
      dk3d_chain_pass_t *pass     = &chain->passes[i];

      pass->scale         = sp->fbo;
      pass->filter        = sp->filter;
      pass->wrap          = sp->wrap;
      pass->mipmap        = sp->mipmap;
      pass->float_fbo     = (sp->fbo.flags & FBO_SCALE_FLAG_FP_FBO) != 0;
      pass->frame_count_mod = sp->frame_count_mod;

      if (!dk3d_compile_slang_pass(sp->source.path,
               &pass->vsh, &pass->fsh, &pass->shader_mem,
               chain->device, chain->uam_initialized))
      {
         RARCH_ERR("[deko3d shader] Pass %u compile failed: %s\n",
               i, sp->source.path);
         dk3d_filter_chain_free(chain);
         return NULL;
      }

      /* Parse UBO + Push block layouts from transformed source. */
      {
         int64_t src_len = 0;
         char *raw = NULL;

         if (filestream_read_file(sp->source.path,
                  (void **)&raw, &src_len) > 0 && src_len > 0)
         {
            char base_dir[PATH_MAX_LENGTH];
            char *resolved, *xformed;

            fill_pathname_basedir(base_dir, sp->source.path,
                  sizeof(base_dir));
            resolved = dk3d_resolve_includes(raw, base_dir, 0);
            free(raw);
            dk3d_strip_slang_pragmas(resolved);
            xformed = dk3d_slang_transform(resolved);
            free(resolved);
            dk3d_parse_pass_uniforms(xformed, pass, shader, chain);
            dk3d_parse_pass_textures(xformed, pass, chain, i);
            free(xformed);
         }
      }

      RARCH_LOG("[deko3d shader] Pass %u compiled: %s (alias: %s, ubo: %u, push: %u, tex: %u)\n",
            i, sp->source.path, *pass->alias ? pass->alias : "(none)",
            pass->num_ubo_slots, pass->num_push_slots, pass->num_tex_bindings);
   }
   chain->num_passes = i;

   /* Load LUT textures. */
   for (i = 0; i < shader->luts && i < DK3D_CHAIN_MAX_LUTS; i++)
   {
      struct video_shader_lut *sl = &shader->lut[i];
      dk3d_chain_lut_t *lut       = &chain->luts[i];

      strlcpy(lut->id, sl->id, sizeof(lut->id));
      lut->filter = sl->filter;
      lut->wrap   = sl->wrap;
      lut->mipmap = sl->mipmap;

      if (!dk3d_load_lut_image(chain->device, chain->queue,
               sl->path, lut))
      {
         RARCH_ERR("[deko3d shader] LUT load failed: %s\n", sl->path);
         dk3d_filter_chain_free(chain);
         return NULL;
      }
      RARCH_LOG("[deko3d shader] LUT loaded: %s (%s)\n", sl->id, sl->path);
   }
   chain->num_luts = i;

   /* Allocate initial FBOs for non-final passes. */
   for (i = 0; i + 1 < chain->num_passes; i++)
   {
      dk3d_chain_pass_t *pass = &chain->passes[i];
      unsigned w = 0, h = 0;

      if (!(pass->scale.flags & FBO_SCALE_FLAG_VALID))
      {
         pass->scale.type_x  = RARCH_SCALE_INPUT;
         pass->scale.type_y  = RARCH_SCALE_INPUT;
         pass->scale.scale_x = 1.0f;
         pass->scale.scale_y = 1.0f;
         pass->scale.flags  |= FBO_SCALE_FLAG_VALID;
      }

      dk3d_compute_pass_fbo_size(&pass->scale,
            info->max_input_width, info->max_input_height,
            info->swapchain_width, info->swapchain_height,
            &w, &h);

      if (!dk3d_chain_create_pass_fbo(chain->device, chain->queue, pass, w, h))
      {
         RARCH_ERR("[deko3d shader] FBO alloc failed for pass %u (%ux%u)\n",
               i, w, h);
         dk3d_filter_chain_free(chain);
         return NULL;
      }
      RARCH_LOG("[deko3d shader] Pass %u FBO: %ux%u\n", i, w, h);
   }

   dk3d_chain_rebuild_descs(chain);

   RARCH_LOG("[deko3d shader] Filter chain ready: %u passes, %u LUTs\n",
         chain->num_passes, chain->num_luts);

   return chain;
}

/* ====================================================================== *
 *  Public API: teardown
 * ====================================================================== */

void dk3d_filter_chain_free(dk3d_filter_chain_t *chain)
{
   unsigned i;
   if (!chain)
      return;

   /* Wait GPU idle before destroying resources. */
   if (chain->queue)
      dkQueueWaitIdle(chain->queue);

   for (i = 0; i < chain->num_passes; i++)
   {
      dk3d_chain_pass_t *p = &chain->passes[i];
      if (p->fbo.memblock)
         dk3d_destroy_image(&p->fbo);
      if (p->fbo_prev.memblock)
         dk3d_destroy_image(&p->fbo_prev);
      if (p->shader_mem)
         dkMemBlockDestroy(p->shader_mem);
   }

   for (i = 0; i < chain->num_luts; i++)
   {
      if (chain->luts[i].image.memblock)
         dk3d_destroy_image(&chain->luts[i].image);
   }

   if (chain->sampler_desc_mem)
      dkMemBlockDestroy(chain->sampler_desc_mem);
   if (chain->image_desc_mem)
      dkMemBlockDestroy(chain->image_desc_mem);
   if (chain->vbo_mem)
      dkMemBlockDestroy(chain->vbo_mem);
   if (chain->global_mem)
      dkMemBlockDestroy(chain->global_mem);
   if (chain->push_mem)
      dkMemBlockDestroy(chain->push_mem);

   if (chain->uam_initialized)
      uam_deinit();

   if (chain->shader)
      free(chain->shader);

   free(chain);
}

/* ====================================================================== *
 *  Public API: per-frame state setters
 * ====================================================================== */

void dk3d_filter_chain_set_input_texture(dk3d_filter_chain_t *chain,
      const DkImage *image, unsigned width, unsigned height)
{
   if (!chain) return;
   chain->input_image  = image;
   chain->input_width  = width;
   chain->input_height = height;
}

void dk3d_filter_chain_set_frame_count(dk3d_filter_chain_t *chain,
      uint64_t count)
{
   if (chain) chain->frame_count = count;
}

void dk3d_filter_chain_set_frame_direction(dk3d_filter_chain_t *chain,
      int32_t direction)
{
   if (chain) chain->frame_direction = direction;
}

void dk3d_filter_chain_set_mvp(dk3d_filter_chain_t *chain,
      const float *mvp)
{
   if (!chain || !mvp) return;
   memcpy(chain->mvp, mvp, sizeof(chain->mvp));
}

void dk3d_filter_chain_set_viewport(dk3d_filter_chain_t *chain,
      unsigned vp_x, unsigned vp_y,
      unsigned vp_width, unsigned vp_height)
{
   if (!chain) return;
   chain->vp_x      = vp_x;
   chain->vp_y      = vp_y;
   chain->vp_width  = vp_width;
   chain->vp_height = vp_height;
}

bool dk3d_filter_chain_update_swapchain(dk3d_filter_chain_t *chain,
      unsigned width, unsigned height, DkImageFormat format)
{
   unsigned i;
   if (!chain) return false;
   if (chain->swapchain_width == width && chain->swapchain_height == height)
      return true;

   chain->swapchain_width  = width;
   chain->swapchain_height = height;

   /* Resize viewport-relative FBOs. */
   for (i = 0; i + 1 < chain->num_passes; i++)
   {
      dk3d_chain_pass_t *pass = &chain->passes[i];
      if (pass->scale.type_x == RARCH_SCALE_VIEWPORT
       || pass->scale.type_y == RARCH_SCALE_VIEWPORT)
      {
         unsigned w = 0, h = 0;
         dk3d_compute_pass_fbo_size(&pass->scale,
               chain->input_width, chain->input_height,
               width, height, &w, &h);
         if (!dk3d_chain_create_pass_fbo(chain->device, chain->queue, pass, w, h))
            return false;
      }
   }

   return true;
}

struct video_shader *dk3d_filter_chain_get_preset(
      dk3d_filter_chain_t *chain)
{
   return chain ? chain->shader : NULL;
}

/* ====================================================================== *
 *  Rendering: build passes
 * ====================================================================== */

static void dk3d_make_view(DkImageView *view, const DkImage *img)
{
   dkImageViewDefaults(view, img);
}

/* Render a single fullscreen-quad pass. */
static void dk3d_render_pass(dk3d_filter_chain_t *chain, DkCmdBuf cmd,
      dk3d_chain_pass_t *pass, unsigned pass_idx,
      const DkImage *src, unsigned src_w, unsigned src_h,
      const DkImage *dst, unsigned dst_w, unsigned dst_h,
      bool is_final)
{
   const DkShader *shaders[2];
   DkRasterizerState rs;
   DkColorState cs;
   DkColorWriteState cws;
   DkDepthStencilState ds;
   DkBlendState bs;
   DkViewport vp;
   DkScissor sc;
   DkImageView dst_view;
   DkImageDescriptor *img_descs;
   /* Per-frame-in-flight heap slot: each in-flight frame writes/binds its own
    * copy of the UBO/push/image-descriptor pools (no cross-frame clobber). */
   unsigned frame_slot   = (unsigned)(chain->frame_count % DK3D_CHAIN_MAX_FRAMES);
   uint32_t global_fbase = frame_slot * (DK3D_UBO_MAX_SIZE  * DK3D_CHAIN_MAX_PASSES);
   uint32_t push_fbase   = frame_slot * (DK3D_PUSH_MAX_SIZE * DK3D_CHAIN_MAX_PASSES);
   uint32_t img_fbase    = frame_slot * (DK3D_CHAIN_MAX_PASSES * DK3D_MAX_TEX_BINDINGS);

   /* Bind render target. */
   dk3d_make_view(&dst_view, dst);
   {
      const DkImageView *targets[1] = { &dst_view };
      dkCmdBufBindRenderTargets(cmd, targets, 1, NULL);
   }

   /* Fill global UBO (binding 0) from parsed slot map. */
   {
      uint8_t *buf = (uint8_t *)chain->global_ptr
                   + global_fbase + pass_idx * DK3D_UBO_MAX_SIZE;
      unsigned k;
      for (k = 0; k < pass->num_ubo_slots; k++)
      {
         dk3d_uniform_slot_t *s = &pass->ubo_slots[k];
         switch (s->sem)
         {
            case DK3D_SEM_MVP:
               memcpy(buf + s->offset, chain->mvp, 64);
               break;
            case DK3D_SEM_OUTPUT_SIZE:
               { float v[4] = {(float)dst_w, (float)dst_h,
                               1.0f / dst_w, 1.0f / dst_h};
                 memcpy(buf + s->offset, v, 16); }
               break;
            case DK3D_SEM_FINAL_VIEWPORT_SIZE:
               { float vw = (float)(chain->vp_width  ? chain->vp_width  : dst_w);
                 float vh = (float)(chain->vp_height ? chain->vp_height : dst_h);
                 float v[4] = { vw, vh, 1.0f / vw, 1.0f / vh };
                 memcpy(buf + s->offset, v, 16); }
               break;
            case DK3D_SEM_SOURCE_SIZE:
               { float v[4] = {(float)src_w, (float)src_h,
                               1.0f / src_w, 1.0f / src_h};
                 memcpy(buf + s->offset, v, 16); }
               break;
            case DK3D_SEM_ORIGINAL_SIZE:
               { float v[4] = {(float)chain->input_width,
                               (float)chain->input_height,
                               1.0f / chain->input_width,
                               1.0f / chain->input_height};
                 memcpy(buf + s->offset, v, 16); }
               break;
            case DK3D_SEM_FRAME_COUNT:
               { uint32_t fc = pass->frame_count_mod
                    ? (uint32_t)(chain->frame_count % pass->frame_count_mod)
                    : (uint32_t)chain->frame_count;
                 memcpy(buf + s->offset, &fc, 4); }
               break;
            case DK3D_SEM_FRAME_DIRECTION:
               memcpy(buf + s->offset, &chain->frame_direction, 4);
               break;
            case DK3D_SEM_PARAM:
               if (s->param_idx >= 0 && chain->shader)
                  memcpy(buf + s->offset,
                        &chain->shader->parameters[s->param_idx].current, 4);
               break;
            case DK3D_SEM_PASS_SIZE:
               if (s->param_idx >= 0
                     && (unsigned)s->param_idx < chain->num_passes)
               {
                  dk3d_chain_pass_t *ref = &chain->passes[s->param_idx];
                  float w = ref->fbo_width  ? (float)ref->fbo_width  : (float)src_w;
                  float h = ref->fbo_height ? (float)ref->fbo_height : (float)src_h;
                  float v[4] = { w, h, 1.0f / w, 1.0f / h };
                  memcpy(buf + s->offset, v, 16);
               }
               break;
            case DK3D_SEM_LUT_SIZE:
               if (s->param_idx >= 0
                     && (unsigned)s->param_idx < chain->num_luts)
               {
                  dk3d_image_t *img = &chain->luts[s->param_idx].image;
                  float w = (float)img->width;
                  float h = (float)img->height;
                  float v[4] = { w, h, 1.0f / w, 1.0f / h };
                  memcpy(buf + s->offset, v, 16);
               }
               break;
         }
      }
   }

   /* Fill push UBO (binding 1) from parsed slot map. */
   {
      uint8_t *buf = (uint8_t *)chain->push_ptr
                   + push_fbase + pass_idx * DK3D_PUSH_MAX_SIZE;
      unsigned k;
      for (k = 0; k < pass->num_push_slots; k++)
      {
         dk3d_uniform_slot_t *s = &pass->push_slots[k];
         switch (s->sem)
         {
            case DK3D_SEM_MVP:
               memcpy(buf + s->offset, chain->mvp, 64);
               break;
            case DK3D_SEM_OUTPUT_SIZE:
               { float v[4] = {(float)dst_w, (float)dst_h,
                               1.0f / dst_w, 1.0f / dst_h};
                 memcpy(buf + s->offset, v, 16); }
               break;
            case DK3D_SEM_FINAL_VIEWPORT_SIZE:
               { float vw = (float)(chain->vp_width  ? chain->vp_width  : dst_w);
                 float vh = (float)(chain->vp_height ? chain->vp_height : dst_h);
                 float v[4] = { vw, vh, 1.0f / vw, 1.0f / vh };
                 memcpy(buf + s->offset, v, 16); }
               break;
            case DK3D_SEM_SOURCE_SIZE:
               { float v[4] = {(float)src_w, (float)src_h,
                               1.0f / src_w, 1.0f / src_h};
                 memcpy(buf + s->offset, v, 16); }
               break;
            case DK3D_SEM_ORIGINAL_SIZE:
               { float v[4] = {(float)chain->input_width,
                               (float)chain->input_height,
                               1.0f / chain->input_width,
                               1.0f / chain->input_height};
                 memcpy(buf + s->offset, v, 16); }
               break;
            case DK3D_SEM_FRAME_COUNT:
               { uint32_t fc = pass->frame_count_mod
                    ? (uint32_t)(chain->frame_count % pass->frame_count_mod)
                    : (uint32_t)chain->frame_count;
                 memcpy(buf + s->offset, &fc, 4); }
               break;
            case DK3D_SEM_FRAME_DIRECTION:
               memcpy(buf + s->offset, &chain->frame_direction, 4);
               break;
            case DK3D_SEM_PARAM:
               if (s->param_idx >= 0 && chain->shader)
                  memcpy(buf + s->offset,
                        &chain->shader->parameters[s->param_idx].current, 4);
               break;
            case DK3D_SEM_PASS_SIZE:
               if (s->param_idx >= 0
                     && (unsigned)s->param_idx < chain->num_passes)
               {
                  dk3d_chain_pass_t *ref = &chain->passes[s->param_idx];
                  float w = ref->fbo_width  ? (float)ref->fbo_width  : (float)src_w;
                  float h = ref->fbo_height ? (float)ref->fbo_height : (float)src_h;
                  float v[4] = { w, h, 1.0f / w, 1.0f / h };
                  memcpy(buf + s->offset, v, 16);
               }
               break;
            case DK3D_SEM_LUT_SIZE:
               if (s->param_idx >= 0
                     && (unsigned)s->param_idx < chain->num_luts)
               {
                  dk3d_image_t *img = &chain->luts[s->param_idx].image;
                  float w = (float)img->width;
                  float h = (float)img->height;
                  float v[4] = { w, h, 1.0f / w, 1.0f / h };
                  memcpy(buf + s->offset, v, 16);
               }
               break;
         }
      }
   }

   /* Bind shaders. */
   shaders[0] = &pass->vsh;
   shaders[1] = &pass->fsh;
   dkCmdBufBindShaders(cmd, DkStageFlag_GraphicsMask, shaders, 2);

   /* Bind fullscreen quad VBO.
    * location 0 = vec4 Position (offset 0, 16 bytes)
    * location 1 = vec2 TexCoord (offset 16, 8 bytes)
    * stride = 24 bytes */
   {
      DkVtxAttribState attribs[2];
      DkVtxBufferState vbuf_state;

      memset(attribs, 0, sizeof(attribs));
      attribs[0].bufferId = 0;
      attribs[0].offset   = 0;
      attribs[0].size     = DkVtxAttribSize_4x32;
      attribs[0].type     = DkVtxAttribType_Float;
      attribs[1].bufferId = 0;
      attribs[1].offset   = 16;
      attribs[1].size     = DkVtxAttribSize_2x32;
      attribs[1].type     = DkVtxAttribType_Float;

      dkCmdBufBindVtxAttribState(cmd, attribs, 2);

      vbuf_state.stride  = 24;
      vbuf_state.divisor = 0;
      dkCmdBufBindVtxBufferState(cmd, &vbuf_state, 1);
      /* Final pass uses the uv-inverted quad (verts 0..3); offscreen passes
       * use the orientation-neutral quad (verts 4..7, byte offset 96). */
      dkCmdBufBindVtxBuffer(cmd, 0,
            chain->vbo_gpu + (is_final ? 0 : (4 * 24)), 4 * 24);
   }

   /* Bind UBOs (per-frame-in-flight base + per-pass offset). */
   {
      DkGpuAddr global_addr = chain->global_gpu + global_fbase
            + pass_idx * DK3D_UBO_MAX_SIZE;
      DkGpuAddr push_addr   = chain->push_gpu + push_fbase
            + pass_idx * DK3D_PUSH_MAX_SIZE;
      dkCmdBufBindUniformBuffer(cmd, DkStage_Vertex, 0, global_addr,
            pass->ubo_size ? pass->ubo_size : 64);
      dkCmdBufBindUniformBuffer(cmd, DkStage_Fragment, 0, global_addr,
            pass->ubo_size ? pass->ubo_size : 64);
      dkCmdBufBindUniformBuffer(cmd, DkStage_Vertex, 1, push_addr,
            pass->push_size ? pass->push_size : DK3D_PUSH_MAX_SIZE);
      dkCmdBufBindUniformBuffer(cmd, DkStage_Fragment, 1, push_addr,
            pass->push_size ? pass->push_size : DK3D_PUSH_MAX_SIZE);
   }

   /* Build image descriptors for this pass's texture inputs.
    * Each pass appends to the shared image descriptor heap (num_images
    * accumulates across passes) so earlier passes' descriptors survive
    * until the whole command buffer executes. */
   img_descs = (DkImageDescriptor *)dkMemBlockGetCpuAddr(chain->image_desc_mem)
             + img_fbase;

   {
      unsigned t;
      uint32_t tex_handles[DK3D_MAX_TEX_BINDINGS];
      unsigned bindings[DK3D_MAX_TEX_BINDINGS];
      unsigned num_bound = 0;

      for (t = 0; t < pass->num_tex_bindings && t < DK3D_MAX_TEX_BINDINGS; t++)
      {
         dk3d_tex_binding_t *tb = &pass->tex_bindings[t];
         const DkImage *tex_img = NULL;
         unsigned sampler_idx   = 0;
         DkImageView view;

         switch (tb->type)
         {
            case DK3D_TEX_SOURCE:
               tex_img     = src;
               sampler_idx = 2 + pass_idx;
               break;
            case DK3D_TEX_ORIGINAL:
               tex_img     = chain->input_image;
               sampler_idx = 1; /* nearest */
               break;
            case DK3D_TEX_PASS_OUTPUT:
               if (tb->index >= 0 && (unsigned)tb->index < chain->num_passes
                     && chain->passes[tb->index].fbo.memblock)
               {
                  tex_img     = &chain->passes[tb->index].fbo.image;
                  sampler_idx = 2 + (unsigned)tb->index;
               }
               break;
            case DK3D_TEX_FEEDBACK:
               /* Last frame's output: the history buffer swapped out of
                * passes[index].fbo at frame start. Falls back to the live
                * FBO on the first frame before fbo_prev holds valid pixels. */
               if (tb->index >= 0 && (unsigned)tb->index < chain->num_passes)
               {
                  dk3d_chain_pass_t *fp = &chain->passes[tb->index];
                  if (fp->fbo_prev.memblock)
                     tex_img = &fp->fbo_prev.image;
                  else if (fp->fbo.memblock)
                     tex_img = &fp->fbo.image;
                  sampler_idx = 2 + (unsigned)tb->index;
               }
               break;
            case DK3D_TEX_LUT:
               if (tb->index >= 0 && (unsigned)tb->index < chain->num_luts)
               {
                  tex_img     = &chain->luts[tb->index].image.image;
                  sampler_idx = 2 + chain->num_passes + (unsigned)tb->index;
               }
               break;
         }

         if (!tex_img)
            continue;

         dk3d_make_view(&view, tex_img);

         /* Force the original game frame to read back opaque alpha. The
          * software and Vulkan paths present an opaque frame; the deko3d
          * HW-direct path exposes the raw PSX framebuffer, whose alpha is the
          * STP/mask bit (~0), which breaks shaders that read Source.a (e.g.
          * crt-guest's "1.0 / LinearizePass.a"). Apply only to the original
          * frame — never the intermediate pass FBOs, which carry real alpha. */
         if (tb->type == DK3D_TEX_ORIGINAL
               || (tb->type == DK3D_TEX_SOURCE && pass_idx == 0))
            view.swizzle[3] = DkImageSwizzle_One;

         dkImageDescriptorInitialize(&img_descs[chain->num_images],
               &view, false, false);
         tex_handles[num_bound] = dkMakeTextureHandle(chain->num_images, sampler_idx);
         bindings[num_bound]    = tb->binding;
         chain->num_images++;
         num_bound++;
      }

      /* Fallback: if no bindings were parsed, bind Source at binding 2. */
      if (num_bound == 0)
      {
         DkImageView view;
         dk3d_make_view(&view, src);
         dkImageDescriptorInitialize(&img_descs[chain->num_images],
               &view, false, false);
         tex_handles[0] = dkMakeTextureHandle(chain->num_images, 2 + pass_idx);
         bindings[0]    = 2;
         chain->num_images++;
         num_bound = 1;
      }

      /* Bind descriptor heaps. */
      dkCmdBufBindSamplerDescriptorSet(cmd, chain->sampler_desc_gpu,
            chain->num_samplers);
      dkCmdBufBindImageDescriptorSet(cmd,
            chain->image_desc_gpu + img_fbase * sizeof(DkImageDescriptor),
            chain->num_images);

      /* Bind each texture at its declared binding slot. */
      for (t = 0; t < num_bound; t++)
         dkCmdBufBindTextures(cmd, DkStage_Fragment,
               bindings[t], &tex_handles[t], 1);
   }

   /* Pipeline state. */
   dkRasterizerStateDefaults(&rs);
   rs.cullMode = DkFace_None;
   dkCmdBufBindRasterizerState(cmd, &rs);

   dkDepthStencilStateDefaults(&ds);
   ds.depthTestEnable  = false;
   ds.depthWriteEnable = false;
   dkCmdBufBindDepthStencilState(cmd, &ds);

   dkColorStateDefaults(&cs);
   dkCmdBufBindColorState(cmd, &cs);

   dkColorWriteStateDefaults(&cws);
   dkColorWriteStateSetMask(&cws, 0, DkColorMask_RGBA);
   dkCmdBufBindColorWriteState(cmd, &cws);

   dkBlendStateDefaults(&bs);
   dkCmdBufBindBlendStates(cmd, 0, &bs, 1);

   /* Viewport / scissor. */
   if (is_final)
   {
      vp.x      = (float)chain->vp_x;
      vp.y      = (float)chain->vp_y;
      vp.width  = (float)chain->vp_width;
      vp.height = (float)chain->vp_height;
   }
   else
   {
      vp.x      = 0.0f;
      vp.y      = 0.0f;
      vp.width  = (float)dst_w;
      vp.height = (float)dst_h;
   }
   vp.near = 0.0f;
   vp.far  = 1.0f;
   dkCmdBufSetViewports(cmd, 0, &vp, 1);

   sc.x      = (uint32_t)vp.x;
   sc.y      = (uint32_t)vp.y;
   sc.width  = (uint32_t)vp.width;
   sc.height = (uint32_t)vp.height;
   dkCmdBufSetScissors(cmd, 0, &sc, 1);

   /* Fullscreen quad via triangle strip (4 vertices from VBO). */
   dkCmdBufDraw(cmd, DkPrimitive_TriangleStrip, 4, 1, 0, 0);

   /* Barrier between passes — fragment writes must complete before next
    * pass reads the FBO as a texture. */
   dkCmdBufBarrier(cmd, DkBarrier_Fragments, DkInvalidateFlags_Image);
}

void dk3d_filter_chain_build_offscreen_passes(dk3d_filter_chain_t *chain,
      DkCmdBuf cmdbuf)
{
   unsigned i;
   const DkImage *src;
   unsigned src_w, src_h;

   if (!chain || !chain->input_image)
      return;

   /* Swap double-buffered feedback surfaces once per frame, before any pass
    * renders. After the swap passes[i].fbo is this frame's render target and
    * passes[i].fbo_prev holds last frame's output (what *Feedback samples).
    * Covers every pass incl. the final one, which renders in
    * build_viewport_pass below. */
   for (i = 0; i < chain->num_passes; i++)
   {
      dk3d_chain_pass_t *p = &chain->passes[i];
      if (p->needs_feedback && p->fbo_prev.memblock)
      {
         dk3d_image_t tmp = p->fbo;
         p->fbo      = p->fbo_prev;
         p->fbo_prev = tmp;
      }
   }

   chain->num_images = 0;

   /* A single-pass chain has no offscreen work — the lone pass renders in
    * build_viewport_pass. The swap above still ran for self-feedback. */
   if (chain->num_passes < 2)
      return;

   src   = chain->input_image;
   src_w = chain->input_width;
   src_h = chain->input_height;

   /* Render all passes except the last (which targets the swapchain). */
   for (i = 0; i + 1 < chain->num_passes; i++)
   {
      dk3d_chain_pass_t *pass = &chain->passes[i];
      unsigned fbo_w = 0, fbo_h = 0;

      dk3d_compute_pass_fbo_size(&pass->scale, src_w, src_h,
            chain->vp_width, chain->vp_height, &fbo_w, &fbo_h);

      if (fbo_w != pass->fbo_width || fbo_h != pass->fbo_height)
         dk3d_chain_create_pass_fbo(chain->device, chain->queue, pass, fbo_w, fbo_h);

      dk3d_render_pass(chain, cmdbuf, pass, i,
            src, src_w, src_h,
            &pass->fbo.image, fbo_w, fbo_h,
            false);

      src   = &pass->fbo.image;
      src_w = fbo_w;
      src_h = fbo_h;
   }
}

/* Render the final pass into its own viewport-sized FBO (no longer straight
 * to the swapchain). This gives the final pass a sampleable output so a
 * *Feedback referencing it (e.g. Pass2Feedback on crt-guest-advanced-fastest)
 * reads a real history buffer, and lets the driver composite the result via
 * the 3D blit (dk3d_present_3d) rather than mixing 2D/3D on the swapchain.
 * The pass renders orientation-NEUTRAL like the offscreen passes; the single
 * display flip lives in the present blit. Retrieve the output with
 * dk3d_filter_chain_get_output(). */
void dk3d_filter_chain_build_viewport_pass(dk3d_filter_chain_t *chain,
      DkCmdBuf cmdbuf, const DkImage *swapchain_image)
{
   dk3d_chain_pass_t *final_pass;
   const DkImage *src;
   unsigned src_w, src_h;
   unsigned out_w, out_h;

   (void)swapchain_image; /* final pass targets its own FBO now */

   if (!chain || chain->num_passes == 0 || !chain->input_image)
      return;

   final_pass = &chain->passes[chain->num_passes - 1];

   if (chain->num_passes == 1)
   {
      chain->num_images = 0;
      src   = chain->input_image;
      src_w = chain->input_width;
      src_h = chain->input_height;
   }
   else
   {
      dk3d_chain_pass_t *prev = &chain->passes[chain->num_passes - 2];
      src   = &prev->fbo.image;
      src_w = prev->fbo_width;
      src_h = prev->fbo_height;
   }

   /* Output size = the on-screen viewport (RetroArch's FinalViewport), which
    * is what crt shaders expect for OutputSize / scanline frequency. */
   out_w = chain->vp_width  ? chain->vp_width  : chain->swapchain_width;
   out_h = chain->vp_height ? chain->vp_height : chain->swapchain_height;

   if (out_w != final_pass->fbo_width || out_h != final_pass->fbo_height
         || !final_pass->fbo.memblock)
      dk3d_chain_create_pass_fbo(chain->device, chain->queue,
            final_pass, out_w, out_h);

   /* is_final=false: neutral quad + full-FBO viewport, identical to an
    * offscreen pass. The present (dk3d_present_3d) applies the display flip. */
   dk3d_render_pass(chain, cmdbuf,
         final_pass,
         chain->num_passes - 1,
         src, src_w, src_h,
         &final_pass->fbo.image,
         out_w, out_h,
         false);
}

/* The final pass's rendered output, to be composited onto the swapchain by
 * the driver (3D blit). NULL until a chain with a built final pass exists. */
const DkImage *dk3d_filter_chain_get_output(dk3d_filter_chain_t *chain)
{
   dk3d_chain_pass_t *final_pass;
   if (!chain || chain->num_passes == 0)
      return NULL;
   final_pass = &chain->passes[chain->num_passes - 1];
   return final_pass->fbo.memblock ? &final_pass->fbo.image : NULL;
}
