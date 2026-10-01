import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { queryClient, queryKeys } from '../lib/api/query'
import { listHistory, clearHistory, deleteHistoryItem, setHistoryStarred, type HistoryItem } from '../lib/api/history'
import { toast } from 'sonner'
import { useTranslation } from 'react-i18next'

export function useHistory(limit?: number, offset?: number) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: [...queryKeys.history, limit, offset],
    queryFn: () => listHistory(limit, offset),
  })
  
  return { 
    data: data || [], 
    isLoading, 
    error, 
    refetch 
  }
}

export function useDeleteHistoryItem() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  
  return useMutation({
    mutationFn: deleteHistoryItem,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.history })
      toast.success(t('history.deleted'))
    },
    onError: (error: Error) => {
      toast.error(t('history.deleteFailed'), { description: error.message })
    },
  })
}

export function useClearHistory() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  
  return useMutation({
    mutationFn: clearHistory,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.history })
      toast.success(t('history.cleared'))
    },
    onError: (error: Error) => {
      toast.error(t('history.clearFailed'), { description: error.message })
    },
  })
}

export function useToggleStarred() {
  const qc = useQueryClient()
  
  return useMutation({
    mutationFn: ({ id, starred }: { id: string; starred: boolean }) => setHistoryStarred(id, starred),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.history })
    },
  })
}
