import type { MetadataRoute } from 'next'
import { getAllPosts } from '@/lib/posts'
import { site } from '@/lib/site'

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    { url: `${site.url}/` },
    { url: `${site.url}/writing` },
    ...getAllPosts().map((post) => ({
      url: `${site.url}/writing/${encodeURIComponent(post.slug)}`,
      lastModified: post.date,
    })),
  ]
}
