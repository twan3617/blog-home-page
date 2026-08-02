export default function Date({ dateString }: { dateString: string }) {
  const date = new globalThis.Date(dateString)
  const formatted = new Intl.DateTimeFormat('en-GB', {
    dateStyle: 'long',
    timeZone: 'UTC',
  }).format(date)

  return <time dateTime={dateString}>{formatted}</time>
}
