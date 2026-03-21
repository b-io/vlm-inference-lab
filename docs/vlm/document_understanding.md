# Document Understanding

## Definition
The ability of VLMs to process, analyze, and extract information from structured and unstructured documents (e.g., PDFs, invoices, forms).

## Core Tasks
- **OCR (Optical Character Recognition)**: Extracting text from images.
- **Form Understanding**: Mapping keys to values (e.g., "Invoice Number: 1234").
- **Table Extraction**: Identifying and converting tables to structured data (e.g., CSV, JSON).
- **VQA (Visual Question Answering)**: Answering questions about the document content.

## Model Examples
- **Donut (Document Understanding Transformer)**: OCR-free end-to-end model.
- **LayoutLM**: Combines text, layout, and image features.
- **LLaVA-Next**: State-of-the-art for high-resolution document processing.

## Challenges
- **Fine-grained Text**: Small text in high-res documents requires high-res vision encoders.
- **Multi-page Reasoning**: Managing context over long PDF documents.
- **Logical Structure**: Understanding reading order and relationships between layout elements.
