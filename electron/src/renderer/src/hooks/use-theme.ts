import { useSyncExternalStore } from 'react'
export type Theme = 'system' | 'light' | 'dark'
function snapshot(): Theme {
  try { const saved = localStorage.getItem('voicestudio.theme'); return saved === 'light' || saved === 'dark' ? saved : 'system' }
  catch { return 'system' }
}
function apply() {
  const theme = snapshot()
  document.documentElement.classList.toggle('dark', theme === 'dark' ||
    (theme === 'system' && matchMedia('(prefers-color-scheme: dark)').matches))
}
function subscribe(listener: () => void) {
  const media = matchMedia('(prefers-color-scheme: dark)')
  const changed = () => { apply(); listener() }
  addEventListener('voicestudio-theme', changed); addEventListener('storage', changed)
  media.addEventListener('change', changed); apply()
  return () => { removeEventListener('voicestudio-theme', changed); removeEventListener('storage', changed); media.removeEventListener('change', changed) }
}
export function useTheme() {
  const theme = useSyncExternalStore(subscribe, snapshot, () => 'system' as Theme)
  function setTheme(value: Theme) {
    localStorage.setItem('voicestudio.theme', value)
    dispatchEvent(new Event('voicestudio-theme'))
  }
  return [theme, setTheme] as const
}
