#!/usr/bin/env python3
"""Check H3 editable PPTX geometry, native charts/media, notes and input fidelity.

Usage: python3 scripts/tests/deck_structure_check.py OUT.pptx [--data DATA.json]
Uses only the Python standard library; does not extract or modify the package.
"""
import argparse
import io
import json
import math
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}
EMU = 914400

def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)

def check(source, data=None, data_path=None):
    errors, summary = [], {}
    def require(condition, message):
        if not condition:
            errors.append(message)
    with zipfile.ZipFile(source) as z:
        names = set(z.namelist())
        require(z.testzip() is None, 'ZIP CRC failure')
        xml = {}
        for name in names:
            if name.endswith(('.xml', '.rels')):
                try:
                    xml[name] = ET.fromstring(z.read(name))
                except ET.ParseError as exc:
                    errors.append(f'{name}: malformed XML: {exc}')
        p = xml.get('ppt/presentation.xml')
        require(p is not None, 'Missing presentation.xml')
        if p is None:
            return errors, summary
        size = p.find('p:sldSz', NS)
        require(size is not None, 'Missing slide size')
        if size is None:
            return errors, summary
        width, height = int(size.get('cx')) / EMU, int(size.get('cy')) / EMU
        require(abs(width - 13.333333) < .002 and abs(height - 7.5) < .002, 'Canvas must be 13.333 × 7.5 inches')
        slides = sorted((n for n in names if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)), key=lambda n: int(re.search(r'(\d+)\.xml', n).group(1)))
        require(10 <= len(slides) <= 14, f'Expected 10–14 slides, got {len(slides)}')
        all_text, objects, tracked = [], 0, 0
        for index, name in enumerate(slides, 1):
            tree = xml[name]
            text = '\n'.join(n.text or '' for n in tree.findall('.//a:t', NS))
            all_text.append(text)
            require(not re.search(r'(?i)\b(?:nan|infinity|undefined|lorem|ipsum)\b', text), f'{name}: invalid visible placeholder')
            require('备注' in text, f'{name}: missing compact source footer')
            note_name = f'ppt/notesSlides/notesSlide{index}.xml'
            require(note_name in xml, f'{name}: missing speaker notes')
            if note_name in xml:
                notes = '\n'.join(n.text or '' for n in xml[note_name].findall('.//a:t', NS))
                require('数据文件：' in notes and '计量：' in notes, f'{name}: notes lack source/measurement definitions')
                marker = 'H3_BOUNDS_JSON='
                require(marker in notes, f'{name}: missing own bounds ledger')
                if marker in notes:
                    try:
                        ledger, _ = json.JSONDecoder().raw_decode(notes.split(marker, 1)[1])
                        tracked += len(ledger)
                        for b in ledger:
                            require(b['x'] >= .495 and b['y'] >= .495 and b['x'] + b['w'] <= width - .495 and b['y'] + b['h'] <= height - .495, f'{name}: ledger outside safe margin: {b["name"]}')
                    except (ValueError, KeyError, TypeError) as exc:
                        errors.append(f'{name}: malformed ledger: {exc}')
            for obj in tree.findall('.//p:spTree/*', NS):
                geom = obj.find('p:spPr/a:xfrm', NS)
                if geom is None:
                    geom = obj.find('p:xfrm', NS)
                if geom is None:
                    continue
                off = geom.find('a:off', NS)
                ext = geom.find('a:ext', NS)
                if off is None or ext is None:
                    continue
                objects += 1
                x, y = int(off.get('x')) / EMU, int(off.get('y')) / EMU
                w, h = int(ext.get('cx')) / EMU, int(ext.get('cy')) / EMU
                require(w >= 0 and h >= 0 and x >= .495 and y >= .495 and x + w <= width - .495 and y + h <= height - .495, f'{name}: actual shape outside margins: {(x, y, w, h)}')
                nv = obj.find('.//p:cNvPr', NS)
                object_name = nv.get('name', '') if nv is not None else ''
                for run in obj.findall('.//a:rPr', NS):
                    if 'sz' in run.attrib:
                        min_size = 9 if object_name.startswith('H3:footer:') else 16
                        require(int(run.get('sz')) >= min_size * 100, f'{name}: {object_name}: text below {min_size}pt')
                    font = run.find('a:latin', NS)
                    if font is not None:
                        require(font.get('typeface') == 'Noto Sans CJK SC', f'{name}: unexpected body font')
        charts = sorted(n for n in names if re.fullmatch(r'ppt/charts/chart\d+\.xml', n))
        chart_values = []
        for name in charts:
            tree = xml[name]
            require(tree.find('.//c:barChart', NS) is not None, f'{name}: expected native bar chart')
            for v in tree.findall('.//c:numCache/c:pt/c:v', NS):
                try:
                    value = float(v.text)
                    require(math.isfinite(value) and value >= 0, f'{name}: invalid chart numeric value')
                    chart_values.append(value)
                except (ValueError, TypeError):
                    errors.append(f'{name}: nonnumeric chart value {v.text!r}')
            external = tree.find('.//c:externalData', NS)
            require(external is not None, f'{name}: missing editable workbook')
        # Validate all internal relationships without following external media links.
        for name, tree in xml.items():
            if not name.endswith('.rels'):
                continue
            parent = posixpath.dirname(posixpath.dirname(name)) if name != '_rels/.rels' else ''
            for rel in tree:
                if rel.get('TargetMode') == 'External':
                    continue
                target = rel.get('Target', '')
                resolved = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(parent, target))
                require(resolved in names, f'{name}: broken relationship {target}')
        full_text = '\n'.join(all_text + [n.text or '' for name in charts for n in xml[name].findall('.//a:t', NS)])
        for method in ['CacheDiT', 'SubBlock', 'AdaLN', 'LoRA', 'VSA']:
            require(method in full_text, f'Missing method: {method}')
        require('不是画质' in full_text and 'profile' in full_text, 'Missing quality/profile caveat')
        videos = [n for n in names if n.startswith('ppt/media/') and n.lower().endswith(('.mp4', '.mov', '.webm', '.m4v'))]
        if data is not None:
            require(data.get('schema_version') == 1, 'Invalid data schema')
            results = data.get('results', {})
            expected = []
            for key in ['cache', 'cache_executor', 'subblock50', 'subblock25', 'larry4', 'light4']:
                row = results.get(key, {})
                expected.extend(v for v in [row.get('cli_s'), row.get('base_cli_s')] if finite(v))
            a = results.get('adaln', {})
            expected.extend(v for v in [a.get('base_cli_s'), *(a.get('samples_s') or []), *(a.get('profile_allocation_s') or [])] if finite(v))
            if not a.get('samples_s') and finite(a.get('cli_s')):
                expected.append(a['cli_s'])
            for value in expected:
                require(any(abs(value - v) <= max(1e-7, abs(value) * 1e-10) for v in chart_values), f'Data measurement absent from native charts: {value}')
            expected_media = 0
            for key in ['larry4', 'light4']:
                item = data.get('media', {}).get(key)
                filename = item if isinstance(item, str) else item.get('path') if isinstance(item, dict) else None
                if filename:
                    file = Path(filename)
                    if not file.is_absolute() and data_path:
                        file = Path(data_path).resolve().parent / file
                    expected_media += file.is_file()
            require(len(videos) >= expected_media, f'Expected {expected_media} embedded videos, got {len(videos)}')
            if not data.get('overnight', {}).get('tasks'):
                require('暂无新增验收结果' in full_text, 'Absent overnight results must be pending, not zero')
        summary = {'slides': len(slides), 'native_charts': len(charts), 'embedded_videos': len(videos), 'actual_drawables': objects, 'tracked_bounds': tracked, 'canvas_inches': [width, height]}
    return errors, summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pptx', help='PowerPoint path, or - for ZIP bytes on stdin')
    parser.add_argument('--data', type=Path)
    options = parser.parse_args()
    data = json.loads(options.data.read_text(encoding='utf-8')) if options.data else None
    source = io.BytesIO(sys.stdin.buffer.read()) if options.pptx == '-' else options.pptx
    try:
        errors, summary = check(source, data, options.data)
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1
    for error in errors:
        print(f'FAIL: {error}', file=sys.stderr)
    print(json.dumps({'passed': not errors, **summary}, ensure_ascii=False))
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
