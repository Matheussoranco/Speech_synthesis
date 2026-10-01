import { useTranslation } from 'react-i18next'

export default function ProfilesWorkspace() {
  const { t } = useTranslation()
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">{t('profiles.title')}</h1>
      <p className="text-text-secondary dark:text-dark-text-secondary">{t('profiles.comingSoon')}</p>
    </div>
  )
}