import { useTranslation } from 'react-i18next'

export default function DesignWorkspace() {
  const { t } = useTranslation()
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">{t('design.title')}</h1>
      <p className="text-text-secondary dark:text-dark-text-secondary">{t('design.comingSoon')}</p>
    </div>
  )
}