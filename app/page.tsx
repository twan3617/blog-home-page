import Link from 'next/link'
import ArticleCard from '@/components/ArticleCard'
import Hero from '@/components/Hero'
import InterestList from '@/components/InterestList'
import Section from '@/components/Section'
import Timeline from '@/components/Timeline'
import { profile } from '@/content/profile'
import { getFeaturedPosts } from '@/lib/posts'

export default function HomePage() {
  const featuredPosts = getFeaturedPosts().slice(0, 4)

  return (
    <main>
      <Hero />
      <Section id="about" eyebrow="Currently" title="A curious life in numbers">
        <InterestList items={profile.current} />
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
        <Timeline entries={profile.experience} />
      </Section>
      <Section id="education" eyebrow="Foundations" title="Mathematics and education">
        <Timeline entries={profile.education} />
      </Section>
      <Section id="beyond" eyebrow="Beyond the screen" title="The rest of life">
        <InterestList items={profile.interests} />
      </Section>
    </main>
  )
}
