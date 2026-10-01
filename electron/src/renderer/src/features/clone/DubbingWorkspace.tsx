import { useTranslation } from 'react-i18next'

interface Props {
  batchMode?: boolean
}

export default function DubbingWorkspace({ batchMode }: Props) {
  const { t } = useTranslation()
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">{batchMode ? t('dubbing.batchTitle') : t('dubbing.title')}</h1>
      <p className="text-text-secondary dark:text-dark-text-secondary">{t('dubbing.comingSoon')}</p>
    </div>
  )
}