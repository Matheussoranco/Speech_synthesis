import { Routes, Route, Navigate } from 'react-router-dom'
import { useEffect, useState } from 'react'

import { useBackendStatus } from './hooks/use-backend-status'
import Layout from './components/Layout'
import CloneWorkspace from './features/clone/CloneWorkspace'
import DesignWorkspace from './features/clone/DesignWorkspace'
import DubbingWorkspace from './features/clone/DubbingWorkspace'
import StoriesWorkspace from './features/clone/StoriesWorkspace'
import AudiobookWorkspace from './features/clone/AudiobookWorkspace'
import GalleryWorkspace from './features/clone/GalleryWorkspace'
import ProfilesWorkspace from './features/clone/ProfilesWorkspace'
import TranscribeWorkspace from './features/clone/TranscribeWorkspace'
import ProjectsWorkspace from './features/clone/ProjectsWorkspace'
import ToolsWorkspace from './features/clone/ToolsWorkspace'
import IntegrationsWorkspace from './features/clone/IntegrationsWorkspace'
import Settings from './routes/Settings'
import LoadingScreen from './components/LoadingScreen'
import ErrorBoundary from './components/ErrorBoundary'

function App() {

  const backendStatus = useBackendStatus()
  const [ready, setReady] = useState(false)

  useEffect(() => {
    if (backendStatus.data?.running) {
      setReady(true)
    }
  }, [backendStatus.data])

  if (!ready) {
    return <LoadingScreen error={backendStatus.error} />
  }

  return (
    <ErrorBoundary>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/clone" replace />} />
          <Route path="/clone" element={<CloneWorkspace />} />
          <Route path="/design" element={<DesignWorkspace />} />
          <Route path="/dubbing" element={<DubbingWorkspace />} />
          <Route path="/dubbing/batch" element={<DubbingWorkspace batchMode />} />
          <Route path="/stories" element={<StoriesWorkspace />} />
          <Route path="/audiobook" element={<AudiobookWorkspace />} />
          <Route path="/gallery" element={<GalleryWorkspace />} />
          <Route path="/profiles" element={<ProfilesWorkspace />} />
          <Route path="/transcribe" element={<TranscribeWorkspace />} />
          <Route path="/projects" element={<ProjectsWorkspace />} />
          <Route path="/tools" element={<ToolsWorkspace />} />
          <Route path="/integrations" element={<IntegrationsWorkspace />} />
          <Route path="/settings/*" element={<Settings />} />
          <Route path="*" element={<Navigate to="/clone" replace />} />
        </Routes>
      </Layout>
    </ErrorBoundary>
  )
}

export default App
