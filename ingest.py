"""
XML Parser and Chunker for US Code Title 26 (Internal Revenue Code)
Parses the XML file and extracts hierarchical chunks with parent/child relationships preserved.
Only extracts: subtitles → chapters → subchapters → parts (optional) → sections → subsections
Skips: editorial notes, repealed sections, and content below subsection level
"""

from lxml import etree
from typing import List, Dict, Optional, Any
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from chunk import chunk_for_rag


@dataclass
class Chunk:
    """Represents a chunk of text with hierarchical metadata."""
    id: str
    text: str
    element_type: str  # e.g., 'section', 'subsection'
    identifier: Optional[str] = None  # e.g., "26 USC § 1"
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}


class XMLParser:
    """Parser for USLM XML documents - Title 26 structure."""
    
    # Elements we want to extract (in hierarchical order)
    STRUCTURAL_ELEMENTS = [
        'title', 'subtitle', 'chapter', 'subchapter', 
        'part', 'subpart', 'section', 'subsection', 
        'paragraph', 'subparagraph', 'clause', 'subclause',
        'item', 'subitem'
    ]
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.chunks: List[Chunk] = []
        self.id_counter = 0
        
    def _extract_table(self, table_elem: etree.Element) -> Dict[str, Any]:
        """Extract table content and convert to structured format."""
        table_data = {
            'headers': [],
            'rows': [],
            'text': ''
        }
        
        # Extract headers from thead (handle namespaces)
        # Find thead element
        thead = None
        for elem in table_elem.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_name == 'thead':
                thead = elem
                break
        
        if thead is not None:
            # Find tr elements within thead (direct children first, then all)
            # Try direct children first
            for tr in thead:
                tr_tag = tr.tag.split('}')[-1] if '}' in tr.tag else tr.tag
                if tr_tag == 'tr':
                    # Find th elements within this tr (direct children only)
                    for th in tr:
                        th_tag = th.tag.split('}')[-1] if '}' in th.tag else th.tag
                        if th_tag == 'th':
                            header_text = self._get_cell_text(th)
                            if header_text:
                                table_data['headers'].append(header_text)
                    # Process all header rows (in case there are multiple)
                    # But typically we only want the first row
                    if table_data['headers']:
                        break
            
            # If we didn't find headers in direct children, try iterating
            if not table_data['headers']:
                for tr in thead.iter():
                    tr_tag = tr.tag.split('}')[-1] if '}' in tr.tag else tr.tag
                    if tr_tag == 'tr':
                        # Find th elements within this tr
                        for th in tr.iter():
                            th_tag = th.tag.split('}')[-1] if '}' in th.tag else th.tag
                            if th_tag == 'th':
                                header_text = self._get_cell_text(th)
                                if header_text and header_text not in table_data['headers']:
                                    table_data['headers'].append(header_text)
                        if table_data['headers']:
                            break
        
        # Extract rows from tbody (handle namespaces)
        tbody = None
        for elem in table_elem.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag_name == 'tbody':
                tbody = elem
                break
        
        if tbody is not None:
            # Find tr elements within tbody
            for tr in tbody.iter():
                tr_tag = tr.tag.split('}')[-1] if '}' in tr.tag else tr.tag
                if tr_tag == 'tr':
                    row = []
                    # Find td elements within this tr (direct children only)
                    for td in tr:
                        td_tag = td.tag.split('}')[-1] if '}' in td.tag else td.tag
                        if td_tag == 'td':
                            cell_text = self._get_cell_text(td)
                            row.append(cell_text)
                    if row:
                        table_data['rows'].append(row)
        
        # Create readable text representation
        text_parts = []
        if table_data['headers']:
            text_parts.append(' | '.join(table_data['headers']))
            text_parts.append('-' * 50)
        
        for row in table_data['rows']:
            text_parts.append(' | '.join(row))
        
        table_data['text'] = '\n'.join(text_parts)
        
        return table_data
    
    def _get_cell_text(self, cell_elem: etree.Element) -> str:
        """Extract text from a table cell (td or th), including all nested elements."""
        # Use itertext() to get all text content regardless of nesting
        # This handles <b>, <span>, <p>, and any other nested elements
        text_parts = []
        
        # Get all text from the element and all its descendants
        for text in cell_elem.itertext():
            cleaned = text.strip()
            if cleaned:
                text_parts.append(cleaned)
        
        # Join all text parts with spaces
        result = ' '.join(text_parts).strip()
        
        # If we got nothing, try a fallback approach
        if not result:
            # Fallback: get direct text and text from direct children
            if cell_elem.text:
                text_parts.append(cell_elem.text.strip())
            
            for child in cell_elem:
                # Recursively get text from child
                child_text = self._get_cell_text(child)
                if child_text:
                    text_parts.append(child_text)
                
                # Add tail text
                if child.tail:
                    text_parts.append(child.tail.strip())
            
            result = ' '.join(text_parts).strip()
        
        return result
    
    def _get_text_content(self, element: etree.Element, skip_notes: bool = True) -> str:
        """Extract text content from an element, skipping notes if requested."""
        text_parts = []
        
        # Get element's own text first
        if element.text:
            text_parts.append(element.text.strip())
        
        for child in element:
            # Skip all note elements (editorial notes, amendments, etc.)
            tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if skip_notes and tag_name == 'note':
                continue
            
            # For sections and subsections, only get content from <content> elements
            # and skip editorial notes, amendments, etc.
            if tag_name == 'content':
                # Get all text from content element
                content_text = self._get_text_content(child, skip_notes=True)
                if content_text:
                    text_parts.append(content_text)
            elif tag_name == 'table':
                # Extract table and include its text representation
                table_data = self._extract_table(child)
                if table_data['text']:
                    # Include table in readable format
                    text_parts.append('\n\n[Table]\n' + table_data['text'] + '\n')
            elif tag_name in ['heading', 'num']:
                # Include headings and numbers
                child_text = self._get_text_content(child, skip_notes=True)
                if child_text:
                    text_parts.append(child_text)
            elif tag_name in ['p', 'paragraph', 'subparagraph', 'clause', 'chapeau', 'continuation']:
                # Include actual content paragraphs
                child_text = self._get_text_content(child, skip_notes=True)
                if child_text:
                    text_parts.append(child_text)
            
            # Add tail text
            if child.tail:
                text_parts.append(child.tail.strip())
        
        return ' '.join(text_parts)
    
    def _get_identifier(self, element: etree.Element) -> Optional[str]:
        """Extract identifier from element (e.g., section number)."""
        # Check for identifier attribute
        identifier = element.get('identifier')
        if identifier:
            # Extract just the section/subsection number from identifier
            # e.g., "/us/usc/t26/s1503" -> "1503"
            # e.g., "/us/usc/t26/s1503/a" -> "1503(a)"
            parts = identifier.split('/')
            if len(parts) > 0:
                last_part = parts[-1]
                if last_part.startswith('s'):
                    # Section identifier
                    section_num = last_part[1:]
                    # Check if there's a subsection
                    if len(parts) > 1 and parts[-2].startswith('s'):
                        # This is a subsection
                        parent_section = parts[-2][1:]
                        subsection = section_num
                        return f"{parent_section}({subsection})"
                    return section_num
                elif last_part.startswith('st'):
                    # Subtitle
                    return last_part[2:]  # e.g., "A"
                elif last_part.startswith('ch'):
                    # Chapter
                    return last_part[2:]  # e.g., "1"
                elif last_part.startswith('sch'):
                    # Subchapter
                    return last_part[3:]  # e.g., "A"
                elif last_part.startswith('pt'):
                    # Part
                    return last_part[2:]  # e.g., "I"
        
        # Check for num element with value
        for num_elem in element.iter():
            tag_name = num_elem.tag.split('}')[-1] if '}' in num_elem.tag else num_elem.tag
            if tag_name == 'num':
                value = num_elem.get('value')
                if value:
                    return value
                # Get text content if no value attribute
                if num_elem.text:
                    return num_elem.text.strip()
                break
        
        return None
    
    def _get_heading(self, element: etree.Element) -> Optional[str]:
        """Extract heading text from element."""
        # Try to find heading element (handle namespaces)
        for heading_elem in element.iter():
            tag_name = heading_elem.tag.split('}')[-1] if '}' in heading_elem.tag else heading_elem.tag
            if tag_name == 'heading':
                heading_text = self._get_text_content(heading_elem, skip_notes=True)
                if heading_text:
                    return heading_text.strip()
        return None
    
    def _is_repealed(self, element: etree.Element) -> bool:
        """Check if an element is repealed by looking for 'Repealed' in heading."""
        heading = self._get_heading(element)
        if heading and 'repealed' in heading.lower():
            return True
        return False
    
    def _generate_id(self, element: etree.Element) -> str:
        """Generate a unique ID for an element."""
        # Use identifier if available
        identifier = self._get_identifier(element)
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        if identifier:
            return f"{tag_name}_{identifier.replace('/', '_')}"
        
        # Use element ID if available
        elem_id = element.get('id')
        if elem_id:
            return elem_id
        
        # Generate sequential ID
        self.id_counter += 1
        return f"chunk_{self.id_counter}"
    
    def _should_chunk(self, element: etree.Element) -> bool:
        """Determine if an element should be chunked."""
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        return tag_name in self.STRUCTURAL_ELEMENTS
    
    def _extract_tables_from_element(self, element: etree.Element) -> List[Dict[str, Any]]:
        """Extract all tables from an element and return structured data."""
        tables = []
        
        # Find all table elements (handle xhtml namespace)
        for table_elem in element.iter():
            tag_name = table_elem.tag.split('}')[-1] if '}' in table_elem.tag else table_elem.tag
            if tag_name == 'table':
                table_data = self._extract_table(table_elem)
                tables.append(table_data)
        
        return tables
    
    def _create_chunk(self, element: etree.Element, parent_chunk: Optional[Chunk]) -> Optional[Chunk]:
        """Create a Chunk object from an element."""
        # Skip if repealed
        if self._is_repealed(element):
            return None
        
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        chunk_id = self._generate_id(element)
        parent_id = parent_chunk.id if parent_chunk else None
        
        # For sections and subsections, extract text content (skip notes)
        # For higher-level elements (subtitle, chapter, etc.), just get heading
        if tag_name in ['section', 'subsection']:
            text = self._get_text_content(element, skip_notes=True)
            # Also extract tables as structured data
            tables = self._extract_tables_from_element(element)
        else:
            # For higher levels, just get heading and identifier
            heading = self._get_heading(element)
            identifier = self._get_identifier(element)
            text = heading or identifier or tag_name
            tables = []
        
        # Skip if no meaningful text
        if not text or len(text.strip()) < 3:
            return None
        
        # Get identifier
        identifier = self._get_identifier(element)
        
        # Get heading
        heading = self._get_heading(element)
        
        # Build metadata
        metadata = {
            'tag': tag_name,
            'element_id': element.get('id'),
            'identifier': identifier,
            'heading': heading,
        }
        
        # Add tables to metadata if present
        if tables:
            metadata['tables'] = tables
        
        # Add any other relevant attributes
        for attr_name, attr_value in element.attrib.items():
            if attr_name not in ['id', 'identifier']:
                metadata[attr_name] = attr_value
        
        chunk = Chunk(
            id=chunk_id,
            text=text,
            element_type=tag_name,
            identifier=identifier,
            parent_id=parent_id,
            metadata=metadata
        )
        
        return chunk
    
    def _process_element(self, element: etree.Element, parent_chunk: Optional[Chunk]):
        """Recursively process an element and its children, skipping notes."""
        tag_name = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        # Skip all note elements entirely (and their children)
        if tag_name == 'note':
            return
        
        # Skip other non-structural elements we don't care about
        # Note: We don't skip 'table' here because we want to extract table content
        skip_tags = ['notes', 'toc', 'layout', 'thead', 'tbody', 'tr', 'td', 'th', 'colgroup', 'col']
        if tag_name in skip_tags:
            return
        
        # Check if we should chunk this element
        if self._should_chunk(element):
            chunk = self._create_chunk(element, parent_chunk)
            if chunk:
                self.chunks.append(chunk)
                parent_chunk = chunk
        
        # Process children (but skip notes and other unwanted elements)
        for child in element:
            child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            # Skip notes and other unwanted elements
            if child_tag not in ['note', 'notes', 'toc', 'layout', 'table']:
                self._process_element(child, parent_chunk)
    
    def parse(self) -> List[Chunk]:
        """Parse the XML file and extract chunks."""
        print(f"Parsing XML file: {self.xml_path}")
        
        # Parse the XML file
        tree = etree.parse(self.xml_path)
        root = tree.getroot()
        
        # Find the main content area
        main = None
        # Try to find main element
        for elem in root.iter():
            if elem.tag.endswith('}main'):
                main = elem
                break
        
        if main is None:
            # If no main element, start from root
            main = root
        
        # Recursively process elements starting from main
        self._process_element(main, None)
        
        print(f"Extracted {len(self.chunks)} chunks")
        return self.chunks
    
    def save_chunks(self, output_path: str, format: str = 'json'):
        """Save chunks to a file."""
        if format == 'json':
            chunks_dict = [asdict(chunk) for chunk in self.chunks]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.chunks)} chunks to {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}")


def main():
    """Main function to parse XML and extract chunks."""
    xml_path = Path(__file__).parent / "usc26.xml"
    output_path = Path(__file__).parent / "chunks.json"
    rag_output_path = Path(__file__).parent / "rag_chunks.json"
    
    # Parse XML
    parser = XMLParser(str(xml_path))
    chunks = parser.parse()
    
    # Save raw chunks
    parser.save_chunks(str(output_path))
    
    # Convert to dictionaries for chunking
    chunks_dict = [asdict(chunk) for chunk in chunks]
    
    # Apply chunking for RAG (split large chunks)
    print("\nApplying chunking for RAG...")
    rag_chunks = chunk_for_rag(chunks_dict, chunk_size=1000)
    
    # Save RAG-ready chunks
    with open(rag_output_path, 'w', encoding='utf-8') as f:
        json.dump(rag_chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(rag_chunks)} RAG-ready chunks to {rag_output_path}")
    
    # Print summary
    print("\n=== Parsing Summary ===")
    print(f"Total raw chunks: {len(chunks)}")
    print(f"Total RAG chunks: {len(rag_chunks)}")
    
    # Count by type
    type_counts = {}
    for chunk in chunks:
        type_counts[chunk.element_type] = type_counts.get(chunk.element_type, 0) + 1
    
    print("\nChunks by type:")
    for elem_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {elem_type}: {count}")
    
    # Show sample chunks
    print("\n=== Sample Raw Chunks ===")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Type: {chunk.element_type}")
        print(f"  Identifier: {chunk.identifier}")
        print(f"  Parent ID: {chunk.parent_id}")
        print(f"  Children: {len(chunk.children_ids)}")
        print(f"  Text preview: {chunk.text[:100]}...")
    
    print("\n=== Sample RAG Chunks ===")
    for i, chunk in enumerate(rag_chunks[:3]):
        print(f"\nRAG Chunk {i+1}:")
        print(f"  ID: {chunk['id']}")
        print(f"  Parent ID: {chunk['metadata'].get('parent_id')}")
        print(f"  Children: {len(chunk['metadata'].get('children_ids', []))}")
        print(f"  Text preview: {chunk['text'][:100]}...")


if __name__ == "__main__":
    main()
