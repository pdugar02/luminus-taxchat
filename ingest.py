"""
XML Parser and Chunker for US Code Title 26 (Internal Revenue Code)
Parses the XML file and extracts hierarchical chunks with parent/child relationships preserved.
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
    element_type: str  # e.g., 'section', 'subsection', 'paragraph'
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
    """Parser for USLM XML documents."""
    
    # USLM namespace
    NS = {'uslm': 'http://xml.house.gov/schemas/uslm/1.0'}
    
    # Elements that represent hierarchical structure
    STRUCTURAL_ELEMENTS = [
        'title', 'subtitle', 'chapter', 'subchapter', 
        'part', 'subpart', 'section', 'subsection', 
        'paragraph', 'subparagraph', 'clause', 'subclause',
        'item', 'subitem'
    ]
    
    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.chunks: List[Chunk] = []
        self.element_stack: List[etree.Element] = []  # Track parent hierarchy
        self.id_counter = 0
        
    def _get_text_content(self, element: etree.Element) -> str:
        """Extract all text content from an element, preserving structure."""
        # Use itertext for cleaner text extraction
        text_parts = []
        for text in element.itertext():
            cleaned = text.strip()
            if cleaned:
                text_parts.append(cleaned)
        
        return ' '.join(text_parts)
    
    def _get_identifier(self, element: etree.Element) -> Optional[str]:
        """Extract identifier from element (e.g., section number)."""
        # Check for identifier attribute
        identifier = element.get('identifier')
        if identifier:
            return identifier
        
        # Check for num element with value
        num_elem = element.find('.//uslm:num', self.NS)
        if num_elem is not None:
            value = num_elem.get('value')
            if value:
                return value
            # Get text content if no value attribute
            if num_elem.text:
                return num_elem.text.strip()
        
        return None
    
    def _get_heading(self, element: etree.Element) -> Optional[str]:
        """Extract heading text from element."""
        heading_elem = element.find('.//uslm:heading', self.NS)
        if heading_elem is not None and heading_elem.text:
            return heading_elem.text.strip()
        return None
    
    def _generate_id(self, element: etree.Element) -> str:
        """Generate a unique ID for an element."""
        # Use identifier if available
        identifier = self._get_identifier(element)
        if identifier:
            return f"{element.tag.split('}')[-1]}_{identifier}"
        
        # Use element ID if available
        elem_id = element.get('id')
        if elem_id:
            return elem_id
        
        # Generate sequential ID
        self.id_counter += 1
        return f"chunk_{self.id_counter}"
    
    def _get_parent_id(self) -> Optional[str]:
        """Get the ID of the current parent element."""
        if self.element_stack:
            parent = self.element_stack[-1]
            return self._generate_id(parent)
        return None
    
    def _should_chunk(self, element: etree.Element) -> bool:
        """Determine if an element should be chunked."""
        tag_name = element.tag.split('}')[-1]
        
        # Always chunk structural elements
        if tag_name in self.STRUCTURAL_ELEMENTS:
            return True
        
        # Chunk paragraphs and other content elements
        if tag_name in ['p', 'paragraph', 'subparagraph']:
            return True
        
        return False
    
    def _create_chunk(self, element: etree.Element) -> Chunk:
        """Create a Chunk object from an element."""
        tag_name = element.tag.split('}')[-1]
        chunk_id = self._generate_id(element)
        parent_id = self._get_parent_id()
        
        # Extract text content
        text = self._get_text_content(element)
        
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
    
    def parse(self) -> List[Chunk]:
        """Parse the XML file and extract chunks."""
        print(f"Parsing XML file: {self.xml_path}")
        
        # Parse the XML file
        tree = etree.parse(self.xml_path)
        root = tree.getroot()
        
        # Register namespace for easier searching
        for prefix, uri in root.nsmap.items():
            if uri == self.NS['uslm']:
                etree.register_namespace(prefix or 'uslm', uri)
        
        # Find the main content area - try different approaches
        main = None
        if root.tag.endswith('}main'):
            main = root
        else:
            # Try to find main element
            for elem in root.iter():
                if elem.tag.endswith('}main'):
                    main = elem
                    break
        
        if main is None:
            # If no main element, start from root
            main = root
        
        # Recursively process elements
        self._process_element(main, None)
        
        print(f"Extracted {len(self.chunks)} chunks")
        return self.chunks
    
    def _process_element(self, element: etree.Element, parent_chunk: Optional[Chunk]):
        """Recursively process an element and its children."""
        # Check if we should chunk this element
        if self._should_chunk(element):
            chunk = self._create_chunk(element)
            
            # Set parent relationship
            if parent_chunk:
                chunk.parent_id = parent_chunk.id
                parent_chunk.children_ids.append(chunk.id)
            
            self.chunks.append(chunk)
            parent_chunk = chunk
        
        # Process children
        for child in element:
            self._process_element(child, parent_chunk)
    
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
