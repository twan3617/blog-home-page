import type { Metadata } from 'next'
import Link from 'next/link'
import { notFound } from 'next/navigation'
import { getPost, getPostSlugs, resolvePostSlug } from '@/lib/posts'
import { site } from '@/lib/site'
import styles from './article.module.css'

type Props = { params: Promise<{ slug: string }> }

const formatter = new Intl.DateTimeFormat('en-GB', {
  dateStyle: 'long',
  timeZone: 'UTC',
})

export const dynamicParams = false

export function generateStaticParams() {
  return getPostSlugs().map((slug) => ({ slug }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug: routeSlug } = await params
  const slug = resolvePostSlug(routeSlug)
  if (!slug) return {}
  const post = await getPost(slug)

  return {
    title: post.title,
    description: post.description,
    alternates: { canonical: `/writing/${encodeURIComponent(slug)}` },
    openGraph: {
      title: post.title,
      description: post.description,
      type: 'article',
      url: `/writing/${encodeURIComponent(slug)}`,
      siteName: site.name,
      locale: site.locale,
      images: [site.socialImage],
    },
    twitter: {
      card: 'summary_large_image',
      title: post.title,
      description: post.description,
      images: [site.socialImage.url],
    },
  }
}

export default async function ArticlePage({ params }: Props) {
  const { slug: routeSlug } = await params
  const slug = resolvePostSlug(routeSlug)
  if (!slug) notFound()
  const post = await getPost(slug)

  return (
    <main className={`${styles.shell} shell`}>
      <article>
        <header className={styles.header}>
          <p className={styles.topics}>{post.topics.join(' · ')}</p>
          <h1>{post.title}</h1>
          <time dateTime={post.date}>{formatter.format(new Date(post.date))}</time>
        </header>
        <div
          className={styles.prose}
          dangerouslySetInnerHTML={{ __html: post.contentHtml }}
        />
        <Link className={styles.back} href="/writing">
          <span aria-hidden="true">←</span> Back to all writing
        </Link>
      </article>
    </main>
  )
}
