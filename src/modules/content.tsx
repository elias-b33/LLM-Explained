import { lazy, type ComponentType, type LazyExoticComponent } from 'react'

/**
 * Maps a module slug to its lazily-loaded lesson component. Code-splitting per
 * module keeps the initial bundle small — heavy pieces (like the tokenizer
 * vocabulary used by a couple of modules) load only when you open them.
 */
export const MODULE_CONTENT: Record<string, LazyExoticComponent<ComponentType>> = {
  orientation: lazy(() => import('./m00-orientation')),
  tokenization: lazy(() => import('./m01-tokenization')),
  embeddings: lazy(() => import('./m02-embeddings')),
  transformer: lazy(() => import('./m03-transformer')),
  attention: lazy(() => import('./m04-attention')),
  layers: lazy(() => import('./m05-layers')),
  sampling: lazy(() => import('./m06-sampling')),
  pretraining: lazy(() => import('./m07-pretraining')),
  'post-training': lazy(() => import('./m08-post-training')),
  inference: lazy(() => import('./m09-inference')),
  capstone: lazy(() => import('./m10-capstone')),
}
