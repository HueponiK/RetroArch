#ifndef SHADER_DEKO3D_H
#define SHADER_DEKO3D_H

#include <stdint.h>
#include <stdbool.h>
#include <deko3d.h>

#include "../common/deko3d_common.h"
#include "../video_shader_parse.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DK3D_CHAIN_MAX_PASSES    GFX_MAX_SHADERS
#define DK3D_CHAIN_MAX_LUTS      GFX_MAX_TEXTURES
#define DK3D_CHAIN_MAX_SAMPLERS  (DK3D_CHAIN_MAX_PASSES + DK3D_CHAIN_MAX_LUTS + 2)

typedef struct dk3d_filter_chain dk3d_filter_chain_t;

struct dk3d_filter_chain_create_info
{
   DkDevice   device;
   DkQueue    queue;
   unsigned   max_input_width;
   unsigned   max_input_height;
   unsigned   swapchain_width;
   unsigned   swapchain_height;
   DkImageFormat swapchain_format;
};

dk3d_filter_chain_t *dk3d_filter_chain_create_from_preset(
      const struct dk3d_filter_chain_create_info *info,
      const char *preset_path);

dk3d_filter_chain_t *dk3d_filter_chain_create_default(
      const struct dk3d_filter_chain_create_info *info,
      bool linear);

void dk3d_filter_chain_free(dk3d_filter_chain_t *chain);

void dk3d_filter_chain_set_input_texture(dk3d_filter_chain_t *chain,
      const DkImage *image, unsigned width, unsigned height);

void dk3d_filter_chain_set_frame_count(dk3d_filter_chain_t *chain,
      uint64_t count);

void dk3d_filter_chain_set_frame_direction(dk3d_filter_chain_t *chain,
      int32_t direction);

void dk3d_filter_chain_set_mvp(dk3d_filter_chain_t *chain,
      const float *mvp);

void dk3d_filter_chain_set_viewport(dk3d_filter_chain_t *chain,
      unsigned vp_x, unsigned vp_y,
      unsigned vp_width, unsigned vp_height);

bool dk3d_filter_chain_update_swapchain(dk3d_filter_chain_t *chain,
      unsigned width, unsigned height, DkImageFormat format);

void dk3d_filter_chain_build_offscreen_passes(dk3d_filter_chain_t *chain,
      DkCmdBuf cmdbuf);

void dk3d_filter_chain_build_viewport_pass(dk3d_filter_chain_t *chain,
      DkCmdBuf cmdbuf, const DkImage *swapchain_image);

/* The final pass's rendered output FBO (viewport-sized). The driver
 * composites this onto the swapchain via the 3D blit. NULL if no output. */
const DkImage *dk3d_filter_chain_get_output(dk3d_filter_chain_t *chain);

struct video_shader *dk3d_filter_chain_get_preset(
      dk3d_filter_chain_t *chain);

#ifdef __cplusplus
}
#endif

#endif
