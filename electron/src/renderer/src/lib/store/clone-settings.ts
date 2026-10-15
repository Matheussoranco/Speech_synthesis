import { createStore, useStore } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

export interface CloneSettings {
  text: string
  language: string
  ref_text: string
  instruct: string
  steps: number
  cfg: number
  speed: number
  t_shift: number
  pos_temp: number
  class_temp: number
  layer_penalty: number
  denoise: boolean
  postprocess: boolean
  duration: number
  show_overrides: boolean
  selected_profile_id: string | null
  auto_play: boolean
}

export const DEFAULT_CLONE_SETTINGS: CloneSettings = {
  text: '',
  language: 'auto',
  ref_text: '',
  instruct: '',
  steps: 30,
  cfg: 0.7,
  speed: 1.0,
  t_shift: 0.0,
  pos_temp: 0.7,
  class_temp: 0.7,
  layer_penalty: 0.0,
  denoise: false,
  postprocess: true,
  duration: 0,
  show_overrides: false,
  selected_profile_id: null,
  auto_play: true,
}

const cloneSettingsStore = createStore<CloneSettings>()(persist(
  () => ({ ...DEFAULT_CLONE_SETTINGS }),
  {
    name: 'voicestudio.clone.settings.v1',
    storage: createJSONStorage(() => localStorage),
    partialize: state => ({ ...state, text: '' }),
  },
))

export function useCloneSettings() { return useStore(cloneSettingsStore); }
export function useCloneSetting<K extends keyof CloneSettings>(key: K) {
  return useStore(cloneSettingsStore, state => state[key]);
}

// Selectors
export const cloneSettingsSelectors = {
  text: (state: CloneSettings) => state.text,
  language: (state: CloneSettings) => state.language,
  ref_text: (state: CloneSettings) => state.ref_text,
  instruct: (state: CloneSettings) => state.instruct,
  steps: (state: CloneSettings) => state.steps,
  cfg: (state: CloneSettings) => state.cfg,
  speed: (state: CloneSettings) => state.speed,
  t_shift: (state: CloneSettings) => state.t_shift,
  pos_temp: (state: CloneSettings) => state.pos_temp,
  class_temp: (state: CloneSettings) => state.class_temp,
  layer_penalty: (state: CloneSettings) => state.layer_penalty,
  denoise: (state: CloneSettings) => state.denoise,
  postprocess: (state: CloneSettings) => state.postprocess,
  duration: (state: CloneSettings) => state.duration,
  show_overrides: (state: CloneSettings) => state.show_overrides,
  selected_profile_id: (state: CloneSettings) => state.selected_profile_id,
  auto_play: (state: CloneSettings) => state.auto_play,
}

// Actions
export function setCloneSetting<K extends keyof CloneSettings>(key: K, value: CloneSettings[K]) {
  cloneSettingsStore.setState((prev) => ({ ...prev, [key]: value }))
}

export function patchCloneSettings(partial: Partial<CloneSettings>) {
  cloneSettingsStore.setState((prev) => ({ ...prev, ...partial }))
}

export function resetCloneOverrides() {
  cloneSettingsStore.setState((prev) => ({
    ...prev,
    steps: DEFAULT_CLONE_SETTINGS.steps,
    cfg: DEFAULT_CLONE_SETTINGS.cfg,
    speed: DEFAULT_CLONE_SETTINGS.speed,
    t_shift: DEFAULT_CLONE_SETTINGS.t_shift,
    pos_temp: DEFAULT_CLONE_SETTINGS.pos_temp,
    class_temp: DEFAULT_CLONE_SETTINGS.class_temp,
    layer_penalty: DEFAULT_CLONE_SETTINGS.layer_penalty,
    denoise: DEFAULT_CLONE_SETTINGS.denoise,
    postprocess: DEFAULT_CLONE_SETTINGS.postprocess,
    duration: DEFAULT_CLONE_SETTINGS.duration,
  }))
}

export function resetCloneSettings() {
  cloneSettingsStore.setState(DEFAULT_CLONE_SETTINGS)
}

export { cloneSettingsStore }
