import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { queryClient, queryKeys } from '../lib/api/query'
import { listProfiles, createProfile, deleteProfile, type Profile, type CreateProfileInput } from '../lib/api/profiles'
import { toast } from 'sonner'
import { useTranslation } from 'react-i18next'

export function useProfiles() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: queryKeys.profiles,
    queryFn: listProfiles,
  })
  
  return { 
    data: data || [], 
    isLoading, 
    error, 
    refetch 
  }
}

export function useCreateProfile() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  
  return useMutation({
    mutationFn: createProfile,
    onSuccess: (newProfile) => {
      qc.invalidateQueries({ queryKey: queryKeys.profiles })
      toast.success(t('profiles.created'), { description: newProfile.name })
    },
    onError: (error: Error) => {
      toast.error(t('profiles.createFailed'), { description: error.message })
    },
  })
}

export function useDeleteProfile() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  
  return useMutation({
    mutationFn: deleteProfile,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.profiles })
      toast.success(t('profiles.deleted'))
    },
    onError: (error: Error) => {
      toast.error(t('profiles.deleteFailed'), { description: error.message })
    },
  })
}
