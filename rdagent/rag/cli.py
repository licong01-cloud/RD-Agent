from __future__ import annotations

import argparse
from pathlib import Path

from rdagent.rag.doc_registry import DocRegistry
from rdagent.rag.factory import get_retriever
from rdagent.rag.index_builder import IndexBuilder
from rdagent.rag.settings import SETTINGS


def _repo_root() -> Path:
    # rdagent/rag/cli.py -> repo root
    return Path(__file__).resolve().parents[2]


def cmd_build(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    storage = SETTINGS.storage_path()
    registry = DocRegistry(storage / "doc_registry.json")

    builder = IndexBuilder(
        repo_root=repo_root,
        chroma_dir=storage / "chroma",
        registry=registry,
        kb_version=SETTINGS.kb_version,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    add_dirs = [Path(p) for p in (args.add_dir or [])]
    add_files = [Path(p) for p in (args.add_file or [])]

    # Resolve relative paths against repo root for convenience.
    add_dirs = [(repo_root / p) if not p.is_absolute() else p for p in add_dirs]
    add_files = [(repo_root / p) if not p.is_absolute() else p for p in add_files]

    docs = None
    if add_dirs or add_files:
        docs = list(builder.iter_source_docs_from_dirs(add_dirs)) + list(builder.iter_source_docs_from_files(add_files))

    builder.build(docs=docs)
    print(f"Index built: collection={builder.collection_name} storage={storage}")


def cmd_query(args: argparse.Namespace) -> None:
    retriever = get_retriever()
    filters = {}
    if args.type:
        filters["type"] = args.type
    if args.doc_format:
        filters["doc_format"] = args.doc_format
    if args.tags:
        filters["tags"] = args.tags

    contexts = retriever.retrieve(args.query, top_k=args.top_k, filters=filters or None)
    for i, c in enumerate(contexts):
        d = c.to_dict()
        print(f"[{i}] score={d['score']:.4f} doc_id={d['doc_id']} source={d['source_path']} chunk={d['chunk_id']} type={d['type']} format={d['doc_format']} tags={d['tags']}")
        print(d["text"][:800])
        print("-" * 80)


def cmd_disable(args: argparse.Namespace) -> None:
    retriever = get_retriever()
    retriever.disable_doc(args.doc_id)
    print(f"disabled: {args.doc_id}")


def cmd_enable(args: argparse.Namespace) -> None:
    retriever = get_retriever()
    retriever.enable_doc(args.doc_id)
    print(f"enabled: {args.doc_id}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rdagent-rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--chunk-size", type=int, default=1200)
    p_build.add_argument("--overlap", type=int, default=200)
    p_build.add_argument("--add-dir", action="append", default=None)
    p_build.add_argument("--add-file", action="append", default=None)
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query")
    p_query.add_argument("query")
    p_query.add_argument("--top-k", type=int, default=6)
    p_query.add_argument("--type", dest="type", default=None)
    p_query.add_argument("--doc-format", dest="doc_format", default=None)
    p_query.add_argument("--tags", nargs="*", default=None)
    p_query.set_defaults(func=cmd_query)

    p_dis = sub.add_parser("disable")
    p_dis.add_argument("doc_id")
    p_dis.set_defaults(func=cmd_disable)

    p_en = sub.add_parser("enable")
    p_en.add_argument("doc_id")
    p_en.set_defaults(func=cmd_enable)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
