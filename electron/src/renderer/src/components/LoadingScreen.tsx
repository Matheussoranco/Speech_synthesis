import { useTranslation } from 'react-i18next'
import { Mic, Loader2 } from 'lucide-react'

export default function LoadingScreen() {
  const { t } = useTranslation()
  
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-surface dark:bg-dark-surface">
      <div className="relative">
        <div className="w-24 h-24 rounded-2xl bg-primary/10 dark:bg-primary/20 flex items-center justify-center">
          <Mic size={40} className="text-primary" aria-hidden="true" />
        </div>
        <Loader2 size={32} className="absolute -top-2 -right-2 text-primary animate-spin" aria-hidden="true" />
      </div>
      <div className="mt-6 text-center">
        <h1 className="text-2xl font-semibold text-text-primary dark:text-dark-text-primary">
          {t('loading.title')}
        </h1>
        <p className="mt-2 text-text-secondary dark:text-dark-text-secondary">
          {t('loading.subtitle')}
        </p>
      </div>
      <div className="mt-8 w-64">
        <div className="h-2 bg-border dark:bg-dark-border rounded-full overflow-hidden">
          <div 
            className="h-full bg-primary rounded-full animate-pulse"
            style={{ width: '60%' }}
            aria-hidden="true"
          />
        </div>
      </div>
    </div>
  )
}
