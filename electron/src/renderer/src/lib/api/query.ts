import { QueryClient } from '@tanstack/react-query'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60, // 1 minute
      gcTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

export const queryKeys = {
  profiles: ['profiles'] as const,
  history: ['history'] as const,
  engines: ['engines'] as const,
  systemInfo: ['system', 'info'] as const,
  profile: (id: string) => ['profiles', id] as const,
  historyItem: (id: string) => ['history', id] as const,
}
