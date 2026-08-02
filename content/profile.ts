export type TimelineEntry = {
  period: string
  title: string
  subtitle: string
  location?: string
  summary: string
  href?: string
  highlights?: readonly string[]
}

export type InterestGroup = {
  title: string
  description: string
}

export type Profile = {
  name: string
  statement: string
  introduction: string
  location: string
  navigation: readonly { label: string; href: `#${string}` }[]
  links: {
    resume: string
    linkedin: string
    github: string
    email: `mailto:${string}`
  }
  current: readonly InterestGroup[]
  experience: readonly TimelineEntry[]
  education: readonly TimelineEntry[]
  interests: readonly InterestGroup[]
}

export const profile = {
  name: 'Tony Wang',
  statement: 'I explore mathematics, computation, and the systems they help us understand.',
  introduction: 'I am a data scientist working in quantitative finance, currently based in London.',
  location: 'London, United Kingdom',
  navigation: [
    { label: 'About', href: '#about' },
    { label: 'Writing', href: '#writing' },
    { label: 'Experience', href: '#experience' },
    { label: 'Education', href: '#education' },
    { label: 'Contact', href: '#contact' },
  ],
  links: {
    resume: '/assets/resume/resume.pdf',
    linkedin: 'https://www.linkedin.com/in/twan3617/',
    github: 'https://github.com/twan3617',
    email: 'mailto:tonywang205@yahoo.com.au',
  },
  current: [
    {
      title: 'Work',
      description: 'Data science in quantitative finance at Coolabah Capital Investments.',
    },
    {
      title: 'Place',
      description: 'Living and working in London after a year in Coolabah’s Sydney office.',
    },
    {
      title: 'Exploring',
      description: 'Mathematics, computation, statistics, and the ideas connecting them.',
    },
  ],
  experience: [
    {
      period: '2025 — Present',
      title: 'Data Scientist',
      subtitle: 'Coolabah Capital Investments',
      location: 'London, United Kingdom',
      summary: 'I joined Coolabah’s Data Science team in Sydney, spent my first year there, and have since moved to the London office.',
    },
    {
      period: 'February 2024 — 2025',
      title: 'Contributor Data Analyst',
      subtitle: 'Quantium',
      summary: 'After completing the graduate program, I worked in Quantium’s Health vertical on business-development opportunities in the UK health and pharmaceutical industries.',
    },
    {
      period: 'February 2023 — February 2024',
      title: 'Graduate Data Analyst',
      subtitle: 'Quantium',
      summary: 'I worked across product analytics and consulting, including transaction labelling, customer attribution, platform costing, and global insurance projects.',
      highlights: ['SQL', 'Snowflake', 'Spark', 'Python', 'Winner, Quantium 2023 GenAI Hackathon'],
    },
    {
      period: 'August 2021 — March 2022',
      title: 'Machine Learning Research Assistant',
      subtitle: 'University of Sydney',
      summary: 'I helped develop a streaming matrix-profile proof of concept for detecting regime changes in noisy sensor data.',
    },
    {
      period: '2020 — 2022',
      title: 'Mathematics educator',
      subtitle: 'University of Sydney and QED Education',
      summary: 'I taught high-school and university mathematics, an unusually fulfilling chapter of my working life.',
    },
  ],
  education: [
    {
      period: '2021 — 2022',
      title: 'Master of Mathematics with Excellence',
      subtitle: 'UNSW',
      summary: 'Average mark 93; thesis mark 96. Research on model-selection consistency for the Lasso and a root-log regulariser.',
      href: '/assets/thesis/masters_thesis_tw.pdf',
    },
    {
      period: '2016 — 2020',
      title: 'BSc (Advanced), Honours in Pure Mathematics',
      subtitle: 'University of Sydney',
      summary: 'Average mark 92; thesis mark 95. Research on pseudomonotone operators and anisotropic elliptic equations.',
      href: '/assets/thesis/honours_thesis_tw.pdf',
    },
    {
      period: '2018',
      title: 'Academic exchange',
      subtitle: 'University of California, Berkeley',
      summary: 'Studied partial differential equations, probability theory, real analysis, and discrete mathematics.',
    },
  ],
  interests: [
    { title: 'Books', description: 'Reading widely and keeping a growing bookshelf.' },
    { title: 'Volleyball', description: 'Playing whenever London weather and schedules permit.' },
    { title: 'Piano', description: 'Returning happily to the same three pieces.' },
  ],
} as const satisfies Profile
