#!/usr/bin/env python3
"""Verify every relative HTML link, asset and fragment in a built MkDocs site."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.duplicates = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                self.duplicates.append(attrs['id'])
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if attrs.get(key):
                self.links.append(attrs[key])


def check(directory):
    root = Path(directory).resolve()
    pages = {}
    for path in root.rglob('*.html'):
        page = Page()
        page.feed(path.read_text())
        pages[path] = page
    if not pages or root / 'index.html' not in pages:
        raise ValueError('Expected a nonempty built site with index.html')
    errors = []
    count = 0
    for path, page in pages.items():
        for link in page.links:
            target = urlsplit(link)
            if target.scheme or target.netloc or target.path.startswith('/'):
                continue
            count += 1
            dest = (path.parent / unquote(target.path)).resolve() if target.path else path
            if not dest.is_relative_to(root):
                errors.append(f'{path.relative_to(root)}: target escapes site: {link}')
                continue
            if dest.is_dir():
                dest /= 'index.html'
            if not dest.exists():
                errors.append(f'{path.relative_to(root)}: missing target: {link}')
            elif target.fragment and dest in pages and unquote(target.fragment) not in pages[dest].ids:
                errors.append(f'{path.relative_to(root)}: missing anchor: {link}')
        errors.extend(f'{path.relative_to(root)}: duplicate id: {value}' for value in page.duplicates)
    if errors:
        raise ValueError('\n'.join(errors))
    print(f'PASS: {len(pages)} HTML pages; {count} relative targets/fragments; no duplicate IDs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', default='site')
    args = parser.parse_args()
    try:
        check(args.site)
    except ValueError as error:
        parser.exit(1, str(error) + '\n')
