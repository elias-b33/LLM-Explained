import { useMemo } from 'react'
import { useTexture } from '@react-three/drei'
import { RepeatWrapping, SRGBColorSpace, type Texture } from 'three'

/**
 * Photo-scanned CC0 surfaces from Poly Haven (see public/textures/CREDITS.txt).
 * Each set is loaded once; `tile()` hands out cheap clones with their own
 * repeat (clones share the GPU texture, only the UV transform differs).
 */
const T = (id: string, map: string, res = '1k') => `/textures/${id}/${id}_${map}_${res}.jpg`

const URLS = {
  plasterMap: T('plastered_wall_04', 'diff', '2k'),
  plasterNormal: T('plastered_wall_04', 'nor_gl', '2k'),
  plasterRough: T('plastered_wall_04', 'rough', '2k'),
  carpetMap: T('poly_wool_herringbone', 'diff'),
  carpetNormal: T('poly_wool_herringbone', 'nor_gl'),
  carpetRough: T('poly_wool_herringbone', 'rough'),
  linenNormal: T('rough_linen', 'nor_gl'),
  linenRough: T('rough_linen', 'rough'),
  oakMap: T('white_oak_veneer', 'diff'),
  oakNormal: T('white_oak_veneer', 'nor_gl'),
  oakRough: T('white_oak_veneer', 'rough'),
  marbleMap: T('marble_01', 'diff'),
  marbleNormal: T('marble_01', 'nor_gl'),
  marbleRough: T('marble_01', 'rough'),
}

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
  const t = useTexture(URLS)
  return useMemo(() => {
    for (const [k, tex] of Object.entries(t)) {
      tex.wrapS = tex.wrapT = RepeatWrapping
      tex.anisotropy = 8
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
