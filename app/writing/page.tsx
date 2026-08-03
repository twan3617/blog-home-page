import type { Metadata } from 'next'
import ArticleCard from '@/components/ArticleCard'
import { getAllPosts } from '@/lib/posts'
import { site } from '@/lib/site'

const description =
  'Notes on mathematics, probability, computation, and machine learning.'

export const metadata: Metadata = {
  title: 'Writing',
  description,
  alternates: { canonical: '/writing' },
  openGraph: {
    title: 'Writing | Tony Wang',
    description,
    url: '/writing',
    siteName: site.name,
    locale: site.locale,
    images: [site.socialImage],
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Writing | Tony Wang',
    description,
    images: [site.socialImage.url],
  },
}

export default function WritingPage() {
  const posts = getAllPosts()

  return (
    <main className="archive shell">
      <header className="archiveIntro">
        <p className="archiveEyebrow">Notebook</p>
        <h1>Writing</h1>
        <p>
          Explorations in mathematics, probability, computation, and machine
          learning—written to make the ideas clearer by working through them.
        </p>
      </header>
      <h2 className="visuallyHidden">Article archive</h2>
      <div className="articleGrid">
        {posts.map((post) => (
          <ArticleCard key={post.slug} post={post} />
        ))}
      </div>
    </main>
  )
}
