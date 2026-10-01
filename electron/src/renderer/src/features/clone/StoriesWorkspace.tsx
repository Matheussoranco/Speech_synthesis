import { useTranslation } from 'react-i18next'

export default function StoriesWorkspace() {
  const { t } = useTranslation()
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">{t('stories.title')}</h1>
      <p className="text-text-secondary dark:text-dark-text-secondary">{t('stories.comingSoon')}</p>
    </div>
  )
}