/** Lesson-completion state for a module. */
export type ModuleStatus = 'unseen' | 'in-progress' | 'done'

/** Static metadata for one lesson module. Drives routing, the sidebar, and nav. */
export interface ModuleMeta {
  /** stable id, also used as the URL slug */
  id: string
  slug: string
  /** 0-based position in the curriculum */
  index: number
  title: string
  subtitle: string
  /** estimated reading/interaction time in minutes */
  minutes: number
}

/** Whether a demo computes genuine output or is a curated illustration. */
export type DemoStatus = 'real' | 'illustrative'
