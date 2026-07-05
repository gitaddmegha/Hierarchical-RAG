import fitz
from typing import List, Dict, Any
from PIL import Image 
import io

class HierarchialProcessor:
    def __init__(self):
        pass

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        nodes = []
        doc = fitz.open(file_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            parent_id = f"page_{page_num}"

            parent_node = {"id" : parent_id,"parent_id": None,
             "type" : "page" ,
             "content" : page.get_text()}

            nodes.append(parent_node)

            blocks = page.get_text("blocks")
            for i, block in enumerate(blocks):
                text_content = block[4].strip()
                if text_content: 
                    child_node = {
                        "id" : f"{parent_id}_text_{i}",
                        "parent_id": parent_id,
                        "type" : "text",
                        "content" : text_content
                    }
                    nodes.append(child_node)

            image_list = page.get_images(full= True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                image_node = {
                    "id" : f"{parent_id}_img_{img_index}",
                    "parent_id": parent_id,
                    "type" : "image",
                    "content" : image
                }
                nodes.append(image_node)

        return nodes
