import { ReactNode, useState, useEffect } from 'react'
import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { 
  Mic, 
  Sparkles, 
  Film, 
  BookOpen, 
  Headphones, 
  FolderOpen, 
  User, 
  Settings, 
  Mic2, 
  Briefcase, 
  Wrench, 
  Plug,
  ChevronLeft,
  ChevronRight,
  Moon,
  Sun,
  Monitor,
  Layers,
  Volume2,
  Music,
  Zap,
  Globe,
  Database,
  Cloud,
  Key,
  Shield,
  Bell,
  HelpCircle,
  Menu,
  X,
  Home,
  Layers as LayersIcon,
} from 'lucide-react'
import { useBackendStatus } from '../hooks/use-backend-status'
import { clsx } from 'clsx'

const navigation = [
  { group: 'Voice', items: [
    { path: '/clone', label: 'Clone', icon: Mic, badge: null },
    { path: '/design', label: 'Design', icon: Sparkles, badge: 'New' },
    { path: '/profiles', label: 'Profiles', icon: User, badge: null },
    { path: '/gallery', label: 'Gallery', icon: Headphones, badge: null },
  ]},
  { group: 'Stories', items: [
    { path: '/stories', label: 'Stories', icon: BookOpen, badge: null },
    { path: '/audiobook', label: 'Audiobook', icon: Music, badge: null },
  ]},
  { group: 'Dubbing', items: [
    { path: '/dubbing', label: 'Single', icon: Film, badge: null },
    { path: '/dubbing/batch', label: 'Batch', icon: LayersIcon, badge: null },
  ]},
  { group: 'Tools', items: [
    { path: '/transcribe', label: 'Transcribe', icon: Mic2, badge: null },
    { path: '/projects', label: 'Projects', icon: Briefcase, badge: null },
    { path: '/tools', label: 'Tools', icon: Wrench, badge: null },
    { path: '/integrations', label: 'Integrations', icon: Plug, badge: null },
  ]},
]

function NavItem({ to, label, Icon, badge, isActive, onClick }: { 
  to: string; label: string; Icon: React.ComponentType<{ size?: number }>; badge?: string | null; isActive: boolean; onClick?: () => void 
}) {
  return (
    <NavLink
      to={to}
      onClick={onClick}
      className={({ isActive }) => clsx(
        'flex items-center gap-3 px-3 py-2 rounded-lg transition-colors',
        'text-sm font-medium',
        isActive 
          ? 'bg-primary/10 text-primary dark:bg-primary/20 dark:text-primary' 
          : 'text-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover dark:text-dark-text-secondary',
        'group'
      )}
      title={label}
    >
      <Icon size={18} className={clsx('flex-shrink-0', isActive && 'text-primary')} aria-hidden="true" />
      <span className="flex-1 truncate">{label}</span>
      {badge && (
        <span className={clsx(
          'px-1.5 py-0.5 text-[10px] font-medium rounded-full',
          isActive 
            ? 'bg-primary text-primary-foreground' 
            : 'bg-muted text-muted-foreground'
        )}>
          {badge}
        </span>
      )}
    </NavLink>
  )
}

function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const { t } = useTranslation()
  const location = useLocation()
  const backendStatus = useBackendStatus()
  
  return (
    <aside 
      className={clsx(
        'fixed left-0 top-0 z-40 h-full border-r border-border dark:border-dark-border',
        'transition-all duration-200 ease-out',
        'flex flex-col',
        collapsed ? 'w-16' : 'w-72'
      )}
      style={{ 
        height: 'calc(100vh - 36px)',
        marginTop: '36px' // Account for titlebar
      }}
    >
      {/* Logo/Title area */}
      <div className={clsx(
        'flex items-center justify-between px-3 py-4 border-b border-border dark:border-dark-border',
        'flex-shrink-0'
      )}>
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <Mic size={18} className="text-white" aria-hidden="true" />
            </div>
            <span className="font-semibold text-lg text-text-primary dark:text-dark-text-primary">
              VoiceStudio
            </span>
          </div>
        )}
        <button
          onClick={onToggle}
          className={clsx(
            'p-1.5 rounded-lg transition-colors',
            'hover:bg-surface-hover dark:hover:bg-dark-surface-hover',
            'text-text-muted hover:text-text-primary dark:text-dark-text-muted dark:hover:text-dark-text-primary'
          )}
          aria-label={collapsed ? t('sidebar.expand') : t('sidebar.collapse')}
          aria-expanded={!collapsed}
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-2 space-y-1" role="navigation" aria-label="Main navigation">
        {navigation.map((group) => (
          <div key={group.group} className={clsx('space-y-0.5', !collapsed && 'pr-1')}>
            {!collapsed && (
              <div className="px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-text-muted dark:text-dark-text-muted">
                {t(`nav.${group.group.toLowerCase()}`)}
              </div>
            )}
            {group.items.map((item) => (
              <NavItem
                key={item.path}
                to={item.path}
                label={t(`nav.${item.label.toLowerCase()}`)}
                Icon={item.icon}
                badge={item.badge ? t(`nav.${item.badge.toLowerCase()}`) : null}
                isActive={location.pathname === item.path || (item.path === '/dubbing/batch' && location.pathname.startsWith('/dubbing/batch'))}
              />
            ))}
          </div>
        ))}
      </nav>
      
      {/* Footer - Engine status, Theme, Settings */}
      <div className="border-t border-border dark:border-dark-border p-3 space-y-3">
        {/* Engine Status */}
        <div className={clsx('flex items-center gap-2', collapsed && 'justify-center')}>
          <div className={clsx(
            'w-2 h-2 rounded-full',
            backendStatus.data?.running ? 'bg-success' : 'bg-error'
          )} aria-hidden="true" />
          {!collapsed && (
            <span className={clsx(
              'text-xs font-medium',
              backendStatus.data?.running ? 'text-success' : 'text-error'
            )}>
              {backendStatus.data?.running ? t('status.backend.running') : t('status.backend.stopped')}
            </span>
          )}
        </div>
        
        {/* Theme & Settings row */}
        <div className={clsx('flex items-center gap-1', collapsed && 'justify-center')}>
          <button
            className={clsx(
              'flex-1 p-2 rounded-lg transition-colors',
              'hover:bg-surface-hover dark:hover:bg-dark-surface-hover',
              'text-text-muted hover:text-text-primary dark:hover:text-dark-text-primary'
            )}
            aria-label={t('settings.appearance')}
            title={collapsed ? t('settings.appearance') : ''}
          >
            <Layers size={18} aria-hidden="true" />
          </button>
          <button
            className={clsx(
              'flex-1 p-2 rounded-lg transition-colors',
              'hover:bg-surface-hover dark:hover:bg-dark-surface-hover',
              'text-text-muted hover:text-text-primary dark:hover:text-dark-text-primary'
            )}
            aria-label={t('settings.general')}
            title={collapsed ? t('settings.general') : ''}
          >
            <Settings size={18} aria-hidden="true" />
          </button>
        </div>
        
        {!collapsed && (
          <div className="flex items-center justify-between pt-2">
            <a 
              href="https://github.com/debpalash/VoiceStudio" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-primary dark:hover:text-dark-text-primary transition-colors"
            >
              <Globe size={14} aria-hidden="true" />
              <span>GitHub</span>
            </a>
            <a 
              href="https://discord.gg/bzQavDfVV9" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-primary dark:hover:text-dark-text-primary transition-colors"
            >
              <MessageSquare size={14} aria-hidden="true" />
              <span>Discord</span>
            </a>
          </div>
        )}
      </div>
    </aside>
  )
}

// Need to import MessageSquare
import { MessageSquare } from 'lucide-react'

function Header() {
  const { t } = useTranslation()
  
  return (
    <header className="fixed top-0 left-0 right-0 z-50 h-9 bg-surface/80 dark:bg-dark-surface/80 backdrop-blur-sm border-b border-border dark:border-dark-border flex items-center justify-between px-4 -webkit-app-region: drag">
      <div className="flex items-center gap-4 -webkit-app-region: no-drag">
        {/* Window drag region */}
        <div className="flex-1 -webkit-app-region: drag" />
      </div>
      <div className="flex items-center gap-2 -webkit-app-region: no-drag">
        {/* Theme toggle would go here */}
        <button className="p-1.5 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover text-text-muted hover:text-text-primary dark:hover:text-dark-text-primary transition-colors" aria-label="Theme">
          <Monitor size={18} aria-hidden="true" />
        </button>
      </div>
    </header>
  )
}

export default function Layout({ children }: { children: ReactNode }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  
  return (
    <div className="min-h-screen bg-surface dark:bg-dark-surface">
      <Header />
      <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} />
      <main 
        className={clsx(
          'pt-9 transition-all duration-200',
          sidebarCollapsed ? 'ml-16' : 'ml-72'
        )}
        style={{ minHeight: 'calc(100vh - 36px)' }}
      >
        <div className="p-4 md:p-6 lg:p-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
