import type { Metadata } from 'next'
import type { ReactNode } from 'react'
import { Fraunces, Inter } from 'next/font/google'
import Footer from '@/components/Footer'
import Header from '@/components/Header'
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

export const metadata: Metadata = {
  title: 'Tony Wang',
  description: 'Mathematics, computation, writing, and quantitative finance.',
}

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${serif.variable} ${sans.variable}`}>
        <Header />
        {children}
        <Footer />
      </body>
    </html>
  )
}
