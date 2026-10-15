import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { SavedVoicesPanel } from './CloneWorkspace'
import { useReference, setReferenceFile, clearReference } from '../../lib/store/reference'
import { patchCloneSettings } from '../../lib/store/clone-settings'
import type { Profile } from '../../lib/api/profiles'

export default function ProfilesWorkspace() {
  const { t } = useTranslation()
  const reference = useReference()
  function select(profile: Profile) {
    clearReference()
    patchCloneSettings({ selected_profile_id: profile.id, ref_text: profile.ref_text,
      instruct: profile.instruct, language: profile.language })
  }
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">{t('profiles.title')}</h1>
      <div className="space-y-4">
        <p>{t('profiles.sessionNotice')}</p>
        <label className="block space-y-2">{t('profiles.referenceRequired')}
          <input type="file" accept="audio/*" onChange={event => {
            const file = event.currentTarget.files?.[0]
            if (file) void setReferenceFile(file).then(result => {
              if (!result.ok && !result.cancelled) toast.error(t('clone.invalidAudio'))
            })
            event.currentTarget.value = ''
          }} />
        </label>
        {reference.object_url && <audio controls src={reference.object_url} />}
        <SavedVoicesPanel onSelect={select} />
      </div>
    </div>
  )
}
