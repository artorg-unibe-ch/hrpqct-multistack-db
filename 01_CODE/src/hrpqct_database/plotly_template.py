import plotly.graph_objects as go
import plotly.io as pio

pio.templates["draft"] = go.layout.Template(
    layout_annotations=[
        dict(
            name="draft watermark",
            text="DRAFT",
            textangle=-30,
            opacity=0.1,
            font=dict(color="black", size=100),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    ],
    layout=go.Layout(height=500, width=500),
)


pio.templates["web-personal"] = go.layout.Template(
    # LAYOUT
    layout={
        # Fonts
        # Note - 'family' must be a single string, NOT a list or dict!
        "title": {
            "font": {
                "family": "Computer Modern Sans Serif",
                "size": 30,
                "color": "#333",
            },
            "x": 0.5,
        },
        "font": {"family": "Computer Modern Sans Serif", "size": 16, "color": "#333"},
        # Keep adding others as needed below
        "hovermode": "x unified",
    },
    # DATA
    data={
        # Each graph object must be in a tuple or list for each trace
        "bar": [
            go.Bar(
                texttemplate="%{value:$.2s}",
                textposition="outside",
                textfont={
                    "family": "Computer Modern Sans Serif",
                    "size": 16,
                    "color": "#FFFFFF",
                },
            )
        ]
    },
)
