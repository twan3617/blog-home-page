import type { Metadata } from 'next'
import type { ReactNode } from 'react'
import { Fraunces, Inter } from 'next/font/google'
import Analytics from '@/components/Analytics'
import Footer from '@/components/Footer'
import Header from '@/components/Header'
import { site } from '@/lib/site'
import './globals.css'

const serif = Fraunces({
  subsets: ['latin'],
  variable: '--font-serif',
  display: 'swap',
})

const sans = Inter({
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap',
})

const description =
  'Mathematics, computation, writing, and a professional life in quantitative finance.'

export const metadata: Metadata = {
  metadataBase: new URL(site.url),
  title: {
    default: 'Tony Wang — Mathematics, Computation and Finance',
    template: '%s | Tony Wang',
  },
  description,
  alternates: { canonical: '/' },
  authors: [{ name: site.name, url: site.url }],
  creator: 'Tony Wang',
  icons: { icon: '/favicon.ico' },
  openGraph: {
    title: 'Tony Wang — Mathematics, Computation and Finance',
    description,
    url: '/',
    siteName: site.name,
    locale: site.locale,
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Tony Wang — Mathematics, Computation and Finance',
    description,
    images: [site.socialImage.url],
  },
}

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${serif.variable} ${sans.variable}`}>
        <Header />
        {children}
        <Footer />
        <Analytics />
      </body>
    </html>
  )
}
