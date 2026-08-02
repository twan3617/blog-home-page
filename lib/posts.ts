import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import remarkMath from 'remark-math'
import remarkRehype from 'remark-rehype'
import rehypeRaw from 'rehype-raw'
import rehypeKatex from 'rehype-katex'
import rehypeStringify from 'rehype-stringify'

const postsDirectory = path.join(process.cwd(), 'posts')

export type PostSummary = {
  slug: string
  title: string
  date: string
  description: string
  topics: string[]
  featured: boolean
}

export type Post = PostSummary & {
  contentHtml: string
}

function metadata(fileName: string, data: Record<string, unknown>): Omit<PostSummary, 'slug'> {
  for (const field of ['title', 'description'] as const) {
    if (typeof data[field] !== 'string' || data[field].trim() === '') {
      throw new Error(`${fileName}: ${field} must be a non-empty string`)
    }
  }

  const date = data.date instanceof Date
    ? data.date.toISOString().slice(0, 10)
    : data.date
  if (typeof date !== 'string' || Number.isNaN(Date.parse(date))) {
    throw new Error(`${fileName}: date must be a valid date`)
  }
  if (!Array.isArray(data.topics) || data.topics.some((topic) => typeof topic !== 'string')) {
    throw new Error(`${fileName}: topics must be an array of strings`)
  }
  if (typeof data.featured !== 'boolean') {
    throw new Error(`${fileName}: featured must be a boolean`)
  }

  return {
    title: data.title as string,
    date,
    description: data.description as string,
    topics: data.topics,
    featured: data.featured,
  }
}

export function readPosts(directory = postsDirectory): PostSummary[] {
  return fs
    .readdirSync(directory)
    .filter((fileName) => fileName.endsWith('.md'))
    .map((fileName) => {
      const source = fs.readFileSync(path.join(directory, fileName), 'utf8')
      const { data } = matter(source)
      return {
        slug: fileName.replace(/\.md$/, ''),
        ...metadata(fileName, data),
      }
    })
    .sort((a, b) => b.date.localeCompare(a.date))
}

export function getAllPosts(): PostSummary[] {
  return readPosts()
}

export function getFeaturedPosts(): PostSummary[] {
  return getAllPosts().filter((post) => post.featured)
}

export function getPostSlugs(): string[] {
  return getAllPosts().map(({ slug }) => slug)
}

export async function getPost(slug: string): Promise<Post> {
  const fileName = `${slug}.md`
  const source = fs.readFileSync(path.join(postsDirectory, fileName), 'utf8')
  const { data, content } = matter(source)
  const processed = await unified()
    .use(remarkParse)
    .use(remarkMath)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeKatex)
    .use(rehypeStringify)
    .process(content)

  return {
    slug,
    ...metadata(fileName, data),
    contentHtml: processed.toString(),
  }
}

// Temporary Pages Router compatibility. Removed when /posts/[id] migrates.
export function getSortedPostsData() {
  return getAllPosts().map(({ slug, ...post }) => ({ id: slug, ...post }))
}

export function getAllPostIds() {
  return getPostSlugs().map((id) => ({ params: { id } }))
}

export async function getPostData(id: string) {
  const post = await getPost(id)
  return { id, ...post }
}
