import Link from 'next/link'
import ArticleCard from '@/components/ArticleCard'
import Hero from '@/components/Hero'
import Section from '@/components/Section'
import { getFeaturedPosts } from '@/lib/posts'

export default function HomePage() {
  const featuredPosts = getFeaturedPosts().slice(0, 4)

  return (
    <main>
      <Hero />
      <Section id="about" eyebrow="Currently" title="A curious life in numbers">
        <p className="readingWidth">
          I use mathematics and computation to understand complicated systems. This
          is where I collect the ideas, projects, and questions that stay with me.
        </p>
      </Section>
      <Section id="writing" eyebrow="Selected ideas" title="Writing">
        <div className="articleGrid">
          {featuredPosts.map((post) => (
            <ArticleCard key={post.slug} post={post} />
          ))}
        </div>
        <Link className="textLink" href="/writing">
          View all writing <span aria-hidden="true">→</span>
        </Link>
      </Section>
      <Section id="experience" eyebrow="Professional" title="Experience">
        <p className="readingWidth">A concise career timeline will live here.</p>
      </Section>
      <Section id="education" eyebrow="Foundations" title="Mathematics and education">
        <p className="readingWidth">Research, theses, and education will live here.</p>
      </Section>
      <Section id="beyond" eyebrow="Beyond the screen" title="The rest of life">
        <p className="readingWidth">Books, volleyball, piano, and other interests will live here.</p>
      </Section>
    </main>
  )
}
