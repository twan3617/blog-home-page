import { notFound, permanentRedirect } from 'next/navigation'
import { getPostSlugs, resolvePostSlug } from '@/lib/posts'

type Props = { params: Promise<{ slug: string }> }

export const dynamicParams = false

export function generateStaticParams() {
  return getPostSlugs().map((slug) => ({ slug }))
}

export default async function LegacyPost({ params }: Props) {
  const { slug: routeSlug } = await params
  const slug = resolvePostSlug(routeSlug)
  if (!slug) notFound()
  permanentRedirect(`/writing/${encodeURIComponent(slug)}`)
}
