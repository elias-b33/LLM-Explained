import { createBrowserRouter, Navigate } from 'react-router-dom'
import { AppLayout } from '@/app/AppLayout'
import { Landing } from '@/app/Landing'
import { ModulePage } from '@/app/ModulePage'
import { RouteError } from '@/app/RouteError'

export const router = createBrowserRouter([
  {
    element: <AppLayout />,
    errorElement: <RouteError />,
    children: [
      { path: '/', element: <Landing /> },
      { path: '/module/:slug', element: <ModulePage /> },
      {
        path: '/workshop',
        errorElement: <RouteError />,
        lazy: () => import('@/app/WorkshopPage').then((m) => ({ Component: m.WorkshopPage })),
      },
      {
        path: '/ask',
        errorElement: <RouteError />,
        lazy: () => import('@/app/AskPage').then((m) => ({ Component: m.AskPage })),
      },
    ],
  },
  // Fullscreen, outside the app shell (no sidebar/topbar) and fully lazy.
  {
    path: '/galaxy',
    errorElement: <RouteError />,
    lazy: () => import('@/app/GalaxyPage').then((m) => ({ Component: m.GalaxyPage })),
  },
  {
    path: '/museum/:slug',
    errorElement: <RouteError />,
    lazy: () => import('@/app/MuseumPage').then((m) => ({ Component: m.MuseumPage })),
  },
  { path: '*', element: <Navigate to="/" replace /> },
])
