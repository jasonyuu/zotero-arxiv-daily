from .protocol import Paper
import html


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {
      font-family: Arial, "Microsoft YaHei", "PingFang SC", sans-serif;
      background-color: #ffffff;
      color: #333333;
    }

    .paper-card {
      margin-bottom: 20px;
      border: 1px solid #dddddd;
      border-radius: 10px;
      padding: 18px;
      background-color: #fafafa;
    }

    .paper-title {
      font-size: 19px;
      font-weight: bold;
      color: #222222;
      line-height: 1.5;
    }

    .paper-meta {
      margin-top: 10px;
      font-size: 14px;
      line-height: 1.8;
      color: #555555;
    }

    .paper-meta strong {
      color: #333333;
    }

    .summary {
      margin-top: 12px;
      padding: 12px;
      background-color: #ffffff;
      border-left: 4px solid #337ab7;
      font-size: 14px;
      line-height: 1.8;
      color: #333333;
    }

    .authors {
      margin-top: 8px;
      font-size: 13px;
      line-height: 1.6;
      color: #666666;
    }

    .affiliations {
      margin-top: 4px;
      font-size: 13px;
      line-height: 1.6;
      color: #777777;
    }

    .links {
      margin-top: 14px;
    }

    .button {
      display: inline-block;
      text-decoration: none;
      font-size: 13px;
      font-weight: bold;
      color: #ffffff;
      background-color: #337ab7;
      padding: 7px 14px;
      border-radius: 4px;
      margin-right: 8px;
    }

    .pdf-button {
      background-color: #d9534f;
    }

    .sci-q1 {
      color: #d9534f;
      font-weight: bold;
    }

    .sci-q2 {
      color: #e67e22;
      font-weight: bold;
    }

    .sci-q3 {
      color: #5cb85c;
      font-weight: bold;
    }

    .sci-q4 {
      color: #777777;
      font-weight: bold;
    }

    .sci-unknown {
      color: #999999;
      font-weight: bold;
    }

    .direction {
      color: #337ab7;
      font-weight: bold;
    }

    .relevance {
      color: #666666;
    }
  </style>
</head>

<body>

<div>
    __CONTENT__
</div>

<br><br>

<div style="font-size: 12px; color: #999999;">
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""


def escape(value) -> str:

    if value is None:
        return ""

    return html.escape(
        str(value),
        quote=True
    )


def get_empty_html():

    block_template = """
    <table
        border="0"
        cellpadding="0"
        cellspacing="0"
        width="100%"
        style="
            font-family: Arial, sans-serif;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
            background-color: #f9f9f9;
        "
    >
      <tr>
        <td
          style="
            font-size: 20px;
            font-weight: bold;
            color: #333;
          "
        >
          No Papers Today. Take a Rest!
        </td>
      </tr>
    </table>
    """

    return block_template


def get_sci_text(paper: Paper):

    quartile = getattr(
        paper,
        "sci_quartile",
        None
    )

    if quartile in (1, 2, 3, 4):

        return (
            f'<span class="sci-q{quartile}">'
            f"Q{quartile}"
            f"</span>"
        )

    return (
        '<span class="sci-unknown">'
        "Unknown"
        "</span>"
    )


def get_year_text(paper: Paper):

    year = getattr(
        paper,
        "publication_year",
        None
    )

    if year:

        return escape(year)

    return "Unknown"


def get_journal_text(paper: Paper):

    journal = getattr(
        paper,
        "journal",
        None
    )

    if journal:

        return escape(journal)

    return "Unknown"


def get_direction_text(paper: Paper):

    direction = getattr(
        paper,
        "research_direction",
        None
    )

    if direction:

        return (
            '<span class="direction">'
            f"{escape(direction)}"
            "</span>"
        )

    return "Unknown"


def get_block_html(
    title: str,
    authors: str,
    affiliations: str,
    rate: str,
    year: str,
    journal: str,
    sci: str,
    direction: str,
    tldr: str,
    url: str,
    pdf_url: str
):

    block_template = """
    <div class="paper-card">

      <div class="paper-title">
        {title}
      </div>

      <div class="paper-meta">

        <strong>年份：</strong>
        {year}

        &nbsp;&nbsp;|&nbsp;&nbsp;

        <strong>期刊：</strong>
        {journal}

        <br>

        <strong>SCI分区：</strong>
        {sci}

        &nbsp;&nbsp;|&nbsp;&nbsp;

        <strong>研究方向：</strong>
        {direction}

        <br>

        <strong>相关度：</strong>
        <span class="relevance">
          {rate}
        </span>

      </div>

      <div class="authors">
        <strong>作者：</strong>
        {authors}
      </div>

      <div class="affiliations">
        <strong>作者单位：</strong>
        {affiliations}
      </div>

      <div class="summary">
        <strong>中文简介：</strong>
        {tldr}
      </div>

      <div class="links">

        <a
          href="{url}"
          class="button"
        >
          IEEE Xplore
        </a>

        {pdf_button}

      </div>

    </div>
    """

    if pdf_url:

        pdf_button = f"""
        <a
          href="{pdf_url}"
          class="button pdf-button"
        >
          PDF
        </a>
        """

    else:

        pdf_button = ""

    return block_template.format(
        title=title,
        authors=authors,
        affiliations=affiliations,
        rate=rate,
        year=year,
        journal=journal,
        sci=sci,
        direction=direction,
        tldr=tldr,
        url=url,
        pdf_button=pdf_button
    )


def render_email(
    papers: list[Paper]
) -> str:

    if len(papers) == 0:

        return framework.replace(
            "__CONTENT__",
            get_empty_html()
        )

    parts = []

    for index, p in enumerate(papers, start=1):

        # =====================================================
        # 标题
        # =====================================================

        title = escape(
            p.title
        )

        # =====================================================
        # 作者
        # =====================================================

        author_list = [
            str(a)
            for a in (p.authors or [])
            if a
        ]

        num_authors = len(
            author_list
        )

        if num_authors == 0:

            authors = "Unknown"

        elif num_authors <= 5:

            authors = ", ".join(
                escape(a)
                for a in author_list
            )

        else:

            authors = ", ".join(
                [
                    *[
                        escape(a)
                        for a in author_list[:3]
                    ],
                    "...",
                    *[
                        escape(a)
                        for a in author_list[-2:]
                    ]
                ]
            )

        # =====================================================
        # 作者单位
        # =====================================================

        if p.affiliations:

            affiliations = ", ".join(
                escape(a)
                for a in p.affiliations[:5]
            )

            if len(p.affiliations) > 5:

                affiliations += ", ..."

        else:

            affiliations = "Unknown Affiliation"

        # =====================================================
        # 相关度
        # =====================================================

        if p.score is not None:

            try:

                rate = str(
                    round(
                        float(p.score),
                        1
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                rate = "Unknown"

        else:

            rate = "Unknown"

        # =====================================================
        # 中文简介
        # =====================================================

        if p.tldr:

            tldr = escape(
                p.tldr
            )

        elif p.abstract:

            tldr = escape(
                p.abstract
            )

        else:

            tldr = "暂无简介"

        # =====================================================
        # 链接
        # =====================================================

        url = escape(
            p.url
            if p.url
            else ""
        )

        pdf_url = escape(
            p.pdf_url
            if p.pdf_url
            else ""
        )

        # =====================================================
        # 生成 HTML
        # =====================================================

        parts.append(
            get_block_html(
                title=(
                    f"{index}. {title}"
                ),
                authors=authors,
                affiliations=affiliations,
                rate=rate,
                year=get_year_text(p),
                journal=get_journal_text(p),
                sci=get_sci_text(p),
                direction=get_direction_text(p),
                tldr=tldr,
                url=url,
                pdf_url=pdf_url
            )
        )

    content = (
        "<br>".join(parts)
    )

    return framework.replace(
        "__CONTENT__",
        content
    )