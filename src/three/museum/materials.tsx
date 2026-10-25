import { useMemo } from 'react'
import { useTexture } from '@react-three/drei'
import { RepeatWrapping, SRGBColorSpace, type Texture } from 'three'
import { activeProfile, textureAnisotropy } from './quality'

/**
 * Photo-scanned CC0 surfaces from Poly Haven (see public/textures/CREDITS.txt).
 * Each set is loaded once; `tile()` hands out cheap clones with their own
 * repeat (clones share the GPU texture, only the UV transform differs).
 */
const T = (id: string, map: string, res = '1k') => `/textures/${id}/${id}_${map}_${res}.jpg`

const urls = (res: string) => ({
  plasterMap: T('plastered_wall_04', 'diff', '2k'),
  plasterNormal: T('plastered_wall_04', 'nor_gl', '2k'),
  plasterRough: T('plastered_wall_04', 'rough', '2k'),
  carpetMap: T('poly_wool_herringbone', 'diff', res),
  carpetNormal: T('poly_wool_herringbone', 'nor_gl', res),
  carpetRough: T('poly_wool_herringbone', 'rough', res),
  linenNormal: T('rough_linen', 'nor_gl', res),
  linenRough: T('rough_linen', 'rough', res),
  oakMap: T('white_oak_veneer', 'diff', res),
  oakNormal: T('white_oak_veneer', 'nor_gl', res),
  oakRough: T('white_oak_veneer', 'rough', res),
  marbleMap: T('marble_01', 'diff', res),
  marbleNormal: T('marble_01', 'nor_gl', res),
  marbleRough: T('marble_01', 'rough', res),
})
const URLS_BY_RES: Record<string, ReturnType<typeof urls>> = { '1k': urls('1k'), '2k': urls('2k') }

export interface PbrSet {
  map?: Texture
  normalMap: Texture
  roughnessMap: Texture
}

export interface MuseumTextures {
  plaster: PbrSet
  carpet: PbrSet
  linen: PbrSet
  oak: PbrSet
  marble: PbrSet
}

export function useMuseumTextures(): MuseumTextures {
  // 1K set at Standard, 2K at High (the plaster is 2K in both).
  const t = useTexture(URLS_BY_RES[activeProfile().res])
  return useMemo(() => {
    const aniso = textureAnisotropy()
    for (const [k, tex] of Object.entries(t)) {
      tex.wrapS = tex.wrapT = RepeatWrapping
      tex.anisotropy = aniso
      if (k.endsWith('Map')) tex.colorSpace = SRGBColorSpace
      tex.needsUpdate = true
    }
    return {
      plaster: {
        map: t.plasterMap,
        normalMap: t.plasterNormal,
        roughnessMap: t.plasterRough,
      },
      carpet: {
        map: t.carpetMap,
        normalMap: t.carpetNormal,
        roughnessMap: t.carpetRough,
      },
      linen: { normalMap: t.linenNormal, roughnessMap: t.linenRough },
      oak: { map: t.oakMap, normalMap: t.oakNormal, roughnessMap: t.oakRough },
      marble: {
        map: t.marbleMap,
        normalMap: t.marbleNormal,
        roughnessMap: t.marbleRough,
      },
    }
  }, [t])
}

/** Clone a set with its own repeat. */
export function tile(set: PbrSet, rx: number, ry: number): PbrSet {
  const c = (tex: Texture) => {
    const n = tex.clone()
    n.repeat.set(rx, ry)
    n.needsUpdate = true
    return n
  }
  return {
    map: set.map ? c(set.map) : undefined,
    normalMap: c(set.normalMap),
    roughnessMap: c(set.roughnessMap),
  }
}
