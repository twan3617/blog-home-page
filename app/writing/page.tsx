import type { Metadata } from 'next'
import ArticleCard from '@/components/ArticleCard'
import { getAllPosts } from '@/lib/posts'

export const metadata: Metadata = {
  title: 'Writing | Tony Wang',
  description: 'Notes on mathematics, probability, computation, and machine learning.',
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
