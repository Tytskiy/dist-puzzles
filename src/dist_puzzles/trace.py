from flock import CommunicationLayout, Tracer


def show_trace(trace: Tracer) -> None:
    from IPython.display import SVG, display

    display(SVG(_draw_trace(trace.communication_layout())))


def _label(value: str, x: float, y: float, *, centered: bool = False) -> str:
    anchor = "middle" if centered else "start"
    return f'<text x="{x}" y="{y}" fill="#555" text-anchor="{anchor}">{value}</text>'


def _draw_trace(layout: CommunicationLayout) -> str:
    transfers = layout.transfers
    ranks = layout.world_size
    last = max((transfer.end for transfer in transfers), default=1)
    width, height = max(480, 120 + 150 * last), 60 + 65 * max(ranks, 1)
    spacing = (width - 120) / last
    colors = (
        "#2986BD",
        "#429647",
        "#8C63BD",
        "#D77A22",
        "#D64F69",
        "#17958D",
        "#BA51A1",
        "#6574CF",
        "#A18A22",
        "#A96840",
        "#238FAD",
        "#727E32",
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Arial, sans-serif" font-size="13" '
        'role="img" aria-label="Communication dependencies by rank">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for stage in range(last + 1):
        x = 85 + spacing * stage
        parts.append(_label(str(stage), x, 25, centered=True))
    for rank in range(ranks):
        y = 60 + 65 * rank
        parts.extend(
            [
                _label(f"rank {rank}", 16, y + 5),
                f'<path d="M85 {y} Q{width / 2} {y + 1.5} {width - 25} {y}" '
                'fill="none" stroke="#dedbd6" stroke-linecap="round"/>',
            ]
        )
    for transfer in transfers:
        message = transfer.message
        x1, x2 = 85 + spacing * transfer.start, 85 + spacing * transfer.end
        y1, y2 = 60 + 65 * message.src, 60 + 65 * message.dst
        color = colors[message.src % len(colors)]
        if message.src == message.dst:
            path = f"M{x1} {y1} C{x1 + 60} {y1 - 28} {x2 - 60} {y2 - 28} {x2} {y2}"
            dx, dy = 60.0, 28.0
        else:
            path = f"M{x1} {y1} C{x1 + 57} {y1 - 1.5} {x2 - 62} {y2 + 1} {x2} {y2}"
            dx, dy = 62.0, -1.0
        length = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / length, dy / length
        bx, by = x2 - 8 * ux, y2 - 8 * uy
        parts.extend(
            [
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" '
                'stroke-linecap="round" stroke-linejoin="round">'
                f"<title>rank {message.src} → rank {message.dst}; "
                f"tag={message.tag}; {message.nbytes} B</title></path>",
                f'<path d="M{bx - 3 * uy} {by + 3 * ux} L{x2} {y2} '
                f'L{bx + 3 * uy} {by - 3 * ux}" fill="none" stroke="{color}" stroke-width="2.2" '
                'stroke-linecap="round" stroke-linejoin="round"/>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts)
