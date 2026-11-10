"""PubMed client (via NCBI E-utilities)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from nexus.knowledge.base import KnowledgeClient, KnowledgeError, KnowledgeRecord, KnowledgeSource
from nexus.core.exceptions import PubMedError


class PubMedClient(KnowledgeClient):
    source = KnowledgeSource.PUBMED

    @classmethod
    def default_base_url(cls) -> str:
        return "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    async def fetch(self, record_id: str) -> KnowledgeRecord:
        try:
            response = await self._request(
                "GET",
                "/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": record_id,
                    "rettype": "abstract",
                    "retmode": "xml",
                },
            )
        except Exception as exc:
            raise PubMedError(
                f"PubMed fetch failed for {record_id}: {exc}"
            ) from exc

        records = self._parse_articles(response.text)

        if not records:
            raise PubMedError(f"No article found in PubMed response for {record_id}")

        return records[0]

    async def search(self, query: str, limit: int = 10) -> list[KnowledgeRecord]:
        try:
            response = await self._request(
                "GET",
                "/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": str(limit),
                    "retmode": "json",
                },
            )

            ids = response.json().get("esearchresult", {}).get("idlist", [])

        except Exception as exc:
            raise PubMedError(
                f"PubMed search failed for {query!r}: {exc}"
            ) from exc

        if not ids:
            return []

        try:
            response = await self._request(
                "GET",
                "/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids[:limit]),
                    "rettype": "abstract",
                    "retmode": "xml",
                },
            )

            return self._parse_articles(response.text)

        except Exception as exc:
            raise PubMedError(
                f"PubMed batch fetch failed: {exc}"
            ) from exc

    def _parse_articles(self, xml_text: str) -> list[KnowledgeRecord]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise PubMedError(f"Failed to parse PubMed XML: {exc}") from exc

        records: list[KnowledgeRecord] = []

        for article_node in root.findall(".//PubmedArticle"):

            citation = article_node.find("MedlineCitation")
            article = citation.find("Article") if citation is not None else None

            if citation is None or article is None:
                continue

            pmid = citation.findtext("PMID", "").strip()

            title = article.findtext("ArticleTitle", "").strip() or "(no title)"

            abstract_parts: list[str] = []

            for abstract in article.findall(".//Abstract/AbstractText"):
                label = abstract.get("Label")
                text = "".join(abstract.itertext()).strip()

                if not text:
                    continue

                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)

            abstract_text = " ".join(abstract_parts) or "(no abstract available)"

            authors = []

            for author in article.findall(".//AuthorList/Author"):
                last = author.findtext("LastName", "")
                initials = author.findtext("Initials", "")

                if last:
                    authors.append(
                        f"{last} {initials}".strip()
                    )

            authors_str = ", ".join(authors[:6])

            if len(authors) > 6:
                authors_str += " et al."

            journal = article.findtext(".//Journal/Title", "").strip()

            year = article.findtext(".//PubDate/Year", "").strip()

            volume = article.findtext(".//JournalIssue/Volume", "").strip()

            issue = article.findtext(".//JournalIssue/Issue", "").strip()

            pages = article.findtext(".//Pagination/MedlinePgn", "").strip()

            language = article.findtext("Language", "").strip()

            doi = ""

            pmcid = ""

            for article_id in article_node.findall(".//PubmedData/ArticleIdList/ArticleId"):

                id_type = article_id.attrib.get("IdType", "").lower()

                value = (article_id.text or "").strip()

                if id_type == "doi":
                    doi = value

                elif id_type == "pmc":
                    pmcid = value

            mesh_terms = [
                mesh.findtext("DescriptorName", "").strip()
                for mesh in citation.findall(".//MeshHeading")
                if mesh.find("DescriptorName") is not None
            ]

            publication_types = [
                pt.text.strip()
                for pt in article.findall(".//PublicationType")
                if pt.text
            ]

            affiliations = []

            for aff in article.findall(".//AffiliationInfo/Affiliation"):

                text = "".join(aff.itertext()).strip()

                if text and text not in affiliations:
                    affiliations.append(text)

            records.append(
                KnowledgeRecord(
                    source=self.source,
                    source_id=pmid,
                    title=title,
                    content=abstract_text,
                    uri=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    metadata={
                        "pmid": pmid,
                        "doi": doi,
                        "pmcid": pmcid,
                        "authors": authors_str,
                        "journal": journal,
                        "year": year,
                        "volume": volume,
                        "issue": issue,
                        "pages": pages,
                        "language": language,
                        "mesh_terms": mesh_terms,
                        "publication_types": publication_types,
                        "affiliations": affiliations,
                    },
                )
            )

        return records


__all__ = ["PubMedClient"]


