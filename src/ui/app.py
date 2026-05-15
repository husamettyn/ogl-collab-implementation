"""Streamlit sunum uygulamasi — ogbl-collab Link Prediction."""

from __future__ import annotations

from src.ui.pages import (
    render_algorithm_runner,
    render_dataset_explorer,
    render_graph_explorer,
    render_results_dashboard,
)


def _inject_css() -> None:
    """Minimal CSS for clean presentation."""
    import streamlit as st

    st.markdown(
        """<style>
        /* Streamlit header'i gizle */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        .block-container {
            padding-top: 1rem !important;
        }

        /* Baslik tab bar'in solunda, ayni hizada */
        .stTabs [data-baseweb="tab-list"] {
            padding-left: 110px !important;
        }
        .stTabs [data-baseweb="tab-list"]::before {
            content: "ogbl-collab";
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.9rem;
            font-weight: 700;
            color: #1a1a2e;
        }

        /* Metric kartlari */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: #16213e;
        }
        /* Compact tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1px;
            background: #f0f0f0;
            border-radius: 8px;
            padding: 3px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 5px 14px;
            font-size: 0.82rem;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: #4e79a7 !important;
            color: white !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit presentation app."""
    import streamlit as st

    st.set_page_config(
        page_title="ogbl-collab | Sunum",
        page_icon="🔗",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _inject_css()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Proje Ozeti",
        "Veri Seti Kesfi",
        "Metodoloji",
        "Deney Sonuclari",
        "Sonuc & Tartisma",
        "Interaktif Demo",
    ])

    with tab1:
        _render_overview()

    with tab2:
        render_dataset_explorer()

    with tab3:
        _render_methodology()

    with tab4:
        render_results_dashboard()

    with tab5:
        _render_conclusions()

    with tab6:
        _render_interactive_demo()


def _render_overview() -> None:
    """Proje ozeti ve problem tanimi sayfasi."""
    import streamlit as st

    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        st.markdown("""
### Problem: Akademik Isbirligi Tahmini

Bir arastirmacinin gelecekte kiminle ortak makale yazacagini **gecmis isbirligi verilerine**
bakarak tahmin edebilir miyiz?

Bu bir **link prediction (baglanti tahmini)** problemidir. Amac, verilen bir
akademik isbirligi aginda, henuz isbirligi yapmamis yazar ciftleri arasinda
**gelecekte isbirligi olusma olasiligini** tahmin etmektir.

#### Neden Onemli?

- **Arastirmaci onerme sistemleri**
- **Fon tahsisi ve ekip olusturma**
- **Bilim haritaciligi:** Disiplinler arasi kopruleri ongorme
- **Konferans planlamasi:** Verimli panelist eslestirme
""")

    with col_right:
        st.info(
            "**ogbl-collab**\n\n"
            "- 235,868 arastirmaci\n"
            "- 1,179,052 isbirligi\n"
            "- 128 boyutlu embedding\n"
            "- 1963–2017 arasi veri\n"
            "- Primer metrik: **Hits@50**"
        )

        st.markdown("**Yontemler:**")
        st.markdown("""
| Yontem | Tur |
|--------|-----|
| Common Neighbors | Heuristik baseline |
| MLP | Feature-only ogrenme |
| GCN | Graph Neural Network |
""")

    st.markdown("---")
    st.markdown("### Proje Mimarisi")

    arch_col1, arch_col2, arch_col3 = st.columns(3)

    with arch_col1:
        st.markdown("""
**1. Veri Yukleme**
- OGB paketi
- PyG graf verisi
- Time-based split
- Scale edilebilir train (%10, %50, %100)
""")
    with arch_col2:
        st.markdown("""
**2. Model Egitimi**
- 3 farkli yontem
- Grid-search tuning
- PyTorch + PyG
- CPU/GPU destegi
""")
    with arch_col3:
        st.markdown("""
**3. Degerlendirme**
- OGB Evaluator API
- Hits@10, 50, 100
- Runtime/memory profilleme
- JSON persistans
""")

    st.markdown("---")
    st.markdown("### Komutlar")
    st.code("""# Tek deney
python main.py train --method gcn --scale 1.0 --device cpu

# Benchmark
python main.py benchmark --scales 0.1 0.5 1.0 --assets

# Tuning
python main.py tune --methods mlp gcn --preset quick

# Sunum
streamlit run src/dash.py""", language="bash")


def _render_methodology() -> None:
    """Metodoloji sayfasi."""
    import streamlit as st

    method_tabs = st.tabs([
        "Common Neighbors",
        "MLP",
        "GCN",
        "Karsilastirma",
    ])

    with method_tabs[0]:
        _render_cn_method()

    with method_tabs[1]:
        _render_mlp_method()

    with method_tabs[2]:
        _render_gcn_method()

    with method_tabs[3]:
        _render_method_comparison()


def _render_cn_method() -> None:
    import streamlit as st

    col1, col2 = st.columns([1, 0.7])

    with col1:
        st.markdown("""
### Common Neighbors — Heuristik Baseline

Iki yazarin **ortak collaborator sayisina** bakar.

**Formul:** `score(A, B) = |N(A) ∩ N(B)| + ε(A,B)`

**Fikir:** Ortak tanidik → yuksek isbirligi olasiligi.
""")

    with col2:
        st.success("""
**Avantaj:** Cok hizli, egitimsiz, aciklanabilir
**Dezavantaj:** Feature kullanmaz, soguk baslangic problemi
""")

    st.markdown("""
**Implementasyon:**
- Adjacency map: Hash tabani, O(1) komsu sorgusu
- Optimizasyon: Kucuk kume buyuk kume icinde aranir
- Tie-breaker: Deterministik hash
- Undirected mod: Varsayilan
""")


def _render_mlp_method() -> None:
    import streamlit as st

    col1, col2 = st.columns([1, 0.7])

    with col1:
        st.markdown("""
### MLP — Feature-Only Ogrenme

Node feature'larini kullanan, graf yapisini kullanmayan sinir agi.

**Mimari:** `x_A ⊙ x_B → Linear → ReLU → ... → Sigmoid → score`

**Egitim:** Binary cross-entropy, negatif ornekleme, Adam optimizer.
""")

    with col2:
        st.info("""
**Varsayilan Hiperparametreler:**
- Hidden: 256, Layers: 3
- Dropout: 0.0, LR: 0.01
- Batch: 65K, Epochs: 200
""")

    st.markdown("""
```
Node A (128d) ─┐
               ├ ⊙ → [Linear 128→256] → ReLU → [Linear 256→256] → ReLU → [Linear 256→1] → Sigmoid
Node B (128d) ─┘
```
""")


def _render_gcn_method() -> None:
    import streamlit as st

    col1, col2 = st.columns([1, 0.7])

    with col1:
        st.markdown("""
### GCN — Graph Convolutional Network

Graf yapisini **message passing** ile dogrudan kullanan GNN modeli.

**Uc Bilesen:**
1. **GCN Encoder:** Node feature → komsu agregasyonu → embedding
2. **Hadamard Predictor:** h_A ⊙ h_B → MLP → score
3. **Full-batch message passing**

**Hiperparametreler:**
- Hidden: 256, Layers: 3, Dropout: 0.2
- LR: 0.005, Batch: 65K, Epochs: 200

`h_A^(l+1) = ReLU( W·AGGREGATE({h_B^(l) : B∈N(A)}) )`
""")

    with col2:
        st.warning("""
**Varsayilan Hiperparametreler:**
- Hidden: 256, Layers: 3, Dropout: 0.2
- LR: 0.005, Batch: 65K, Epochs: 200
- Grad clipping: 1.0
""")


def _render_method_comparison() -> None:
    import streamlit as st

    st.markdown("### Yontem Karsilastirmasi")

    st.dataframe({
        "Ozellik": [
            "Graf yapisini kullanir", "Node feature kullanir",
            "Egitim gerektirir", "Aciklanabilirlik",
            "Hiz (scale=1.0)", "Bellek", "Hiperparametre",
        ],
        "CN": ["✅ (yuzeysel)", "❌", "❌", "✅✅✅", "⚡ Saniyeler", "Dusuk", "0"],
        "MLP": ["❌", "✅", "✅ (200 ep)", "✅", "🐢 Dakikalar", "Orta", "5"],
        "GCN": ["✅✅", "✅", "✅ (200 ep)", "⚠️", "🐢🐢 Dakikalar", "Yuksek", "5"],
    }, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Hits@K Metrigi")
    st.markdown("""
Pozitif bir ornegin skorunun, negatifler arasinda **ilk K'ya girme orani**.

`Hits@50 = (Ilk 50'deki pozitif) / (Toplam pozitif)`

- OGB-collab'in resmi metrigi
- Pratik: "En uygun 50 collaborator" onerisi
""")


def _render_conclusions() -> None:
    import streamlit as st

    findings_col1, findings_col2 = st.columns(2)

    with findings_col1:
        st.markdown("### Bulgular")
        st.markdown("""
1. **GCN en yuksek ogrenme kapasitesine sahip.** Graf yapisini kullanarak node feature'larini zenginlestirir. Scale 1.0'da 0.455 Hits@50.

2. **CN sasirtici derecede iyi baseline.** Hizli, egitimsiz, aciklanabilir. Ozellikle scale 1.0'da 0.515 Hits@50.

3. **Scale arttikca tum yontemlerin performansi artar.** Daha fazla train verisi = daha iyi genelleme.

4. **MLP, GCN'den hizli ama daha dusuk performansli.** Feature-only, graf yapisini gormuyor.
""")

    with findings_col2:
        st.markdown("### Trade-off")
        st.markdown("""
| Boyut | Kazanan |
|-------|---------|
| Dogruluk | **GCN** |
| Hiz | **CN** |
| Aciklanabilirlik | **CN** |
| Bellek | **CN** |
| Genellenebilirlik | **GCN** |
""")

    st.markdown("---")
    st.markdown("### Gelecek Calismalar")
    st.markdown("""
- ~~GCN baseline calisiyor~~ ✅ (Hits@50=0.455)
- ~~Multi-seed benchmark~~ 🔄 (calisiyor)
- GAT, GraphSAGE, SEAL gibi gelismis GNN'ler
- Edge weight ve year feature olarak kullanma
- BERT tabanli node embedding'leri
- Ensemble: CN + GCN hibrit skorlama
""")

    st.success("""
**Ozet:** ogbl-collab'da en etkili yontem GCN. Ancak pratikte hiz/aciklanabilirlik 
ihtiyacina gore CN veya MLP tercih edilebilir. Uc yontem tamamlayici bir cozum ailesi olusturur.
""")


def _render_interactive_demo() -> None:
    import streamlit as st

    demo_tab1, demo_tab2 = st.tabs(["Graph Explorer", "Algorithm Runner"])

    with demo_tab1:
        render_graph_explorer()

    with demo_tab2:
        render_algorithm_runner()


if __name__ == "__main__":
    main()
