import { Component, ErrorInfo, ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { AlertTriangle, RefreshCw, X } from 'lucide-react'
import { toast } from 'sonner'

interface Props {
  children: ReactNode
  fallback?: React.ComponentType<{ error: Error; reset: () => void }>
}

interface State {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null }
  
  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }
  
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo)
    toast.error('Something went wrong', {
      description: error.message,
      action: {
        label: 'Reload',
        onClick: () => window.location.reload(),
      },
    })
  }
  
  reset = () => {
    this.setState({ hasError: false, error: null })
  }
  
  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return <this.props.fallback error={this.state.error!} reset={this.reset} />
      }
      
      const { t } = useTranslation()
      const { error } = this.state
      
      return (
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="max-w-md w-full text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-error/10 flex items-center justify-center">
              <AlertTriangle size={32} className="text-error" aria-hidden="true" />
            </div>
            <h2 className="text-xl font-semibold text-text-primary dark:text-dark-text-primary mb-2">
              {t('error.title')}
            </h2>
            <p className="text-text-secondary dark:text-dark-text-secondary mb-6">
              {t('error.description')}
            </p>
            {error && (
              <details className="mb-4 text-left p-3 bg-surface-hover dark:bg-dark-surface-hover rounded-lg border border-border dark:border-dark-border">
                <summary className="font-medium text-text-secondary dark:text-dark-text-secondary cursor-pointer">
                  {t('error.details')}
                </summary>
                <pre className="mt-2 text-xs text-text-muted dark:text-dark-text-muted overflow-auto max-h-40">
                  {error.message}
                  {error.stack && `

${error.stack}`}
                </pre>
              </details>
            )}
            <div className="flex gap-3 justify-center">
              <button
                onClick={this.reset}
                className="px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary-hover transition-colors flex items-center gap-2"
              >
                <RefreshCw size={16} aria-hidden="true" />
                {t('error.tryAgain')}
              </button>
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 border border-border dark:border-dark-border text-text-secondary dark:text-dark-text-secondary rounded-lg font-medium hover:bg-surface-hover dark:hover:bg-dark-surface-hover transition-colors flex items-center gap-2"
              >
                <X size={16} aria-hidden="true" />
                {t('error.reload')}
              </button>
            </div>
          </div>
        </div>
      )
    }
    
    return this.props.children
  }
}
