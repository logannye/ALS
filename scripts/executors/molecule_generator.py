"""De novo molecular generation for ALS drug design targets.

Three generation strategies, all RDKit-only (no PyTorch required):

1. **Scaffold hopping**: Take a known active, keep the core scaffold,
   vary the periphery to improve CNS penetration or reduce toxicity.
2. **Fragment-based growing**: Decompose known actives into BRICS
   fragments, recombine in novel ways.
3. **Matched molecular pair transforms**: Apply known productive
   chemical transformations to candidate molecules.

All candidates are filtered by drug-likeness, CNS MPO (must cross
blood-brain barrier for ALS), and synthetic accessibility.

Usage:
    candidates = generate_candidates(
        target_id="ddt:sigma1r_pocket",
        reference_smiles=["CC1=CC=C(...)"],
        strategy="scaffold_hop",
    )
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GeneratedMolecule:
    """A computationally generated molecule candidate."""
    smiles: str
    parent_smiles: str       # Reference compound it was derived from
    strategy: str            # "scaffold_hop" | "fragment_grow" | "mmp_transform"
    cns_mpo: float = 0.0
    molecular_weight: float = 0.0
    logp: float = 0.0
    lipinski_violations: int = 0
    synthetic_accessibility: float = 10.0
    similarity_to_parent: float = 0.0
    is_novel: bool = True


def generate_candidates(
    target_id: str,
    reference_smiles: list[str],
    strategy: str = "scaffold_hop",
    n_candidates: int = 100,
    filters: Optional[dict] = None,
) -> list[GeneratedMolecule]:
    """Generate novel molecular candidates for a drug design target.

    Parameters
    ----------
    target_id:
        Drug design target ID (e.g. "ddt:sigma1r_pocket").
    reference_smiles:
        SMILES of known active compounds to use as starting points.
    strategy:
        "scaffold_hop", "fragment_grow", or "mmp_transform".
    n_candidates:
        Maximum number of candidates to generate before filtering.
    filters:
        Override default filters: cns_mpo_min, lipinski_max, sa_max, mw_min, mw_max.

    Returns
    -------
    Filtered, scored candidates sorted by CNS MPO (highest first).
    """
    try:
        from rdkit import Chem
    except ImportError:
        logger.error("RDKit not available for molecular generation")
        return []

    if not reference_smiles:
        return []

    default_filters = {
        "cns_mpo_min": 2.5,     # Inclusive — real ChEMBL actives often score 2.5-4.0
        "lipinski_max": 2,      # Allow some violations (many CNS drugs violate 1)
        "sa_max": 7.0,
        "mw_min": 150,
        "mw_max": 600,          # Many S1R ligands are 400-550 MW
    }
    if filters:
        default_filters.update(filters)

    # Parse reference molecules
    ref_mols = []
    for smi in reference_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ref_mols.append((smi, mol))

    if not ref_mols:
        return []

    # Generate raw candidates
    if strategy == "scaffold_hop":
        raw = _scaffold_hop(ref_mols, n_candidates)
    elif strategy == "fragment_grow":
        raw = _fragment_grow(ref_mols, n_candidates)
    elif strategy == "mmp_transform":
        raw = _mmp_transform(ref_mols, n_candidates)
    else:
        raw = _scaffold_hop(ref_mols, n_candidates)  # Default

    # Score and filter
    scored = _score_and_filter(raw, default_filters, ref_mols)
    scored.sort(key=lambda m: m.cns_mpo, reverse=True)

    return scored


# ---------------------------------------------------------------------------
# Strategy A: Scaffold Hopping
# ---------------------------------------------------------------------------

def _scaffold_hop(
    ref_mols: list[tuple[str, "Chem.Mol"]],
    n_candidates: int,
) -> list[tuple[str, str]]:
    """Generate analogs by atom/group substitutions on reference molecules.

    Uses RDKit reaction SMARTS for common medicinal chemistry transformations.
    Returns list of (new_smiles, parent_smiles) tuples.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Reaction SMARTS for bioisosteric replacements (CNS optimization)
    REACTIONS = [
        "[cH:1]>>[c:1]F",                   # Aromatic H → F (metabolic block)
        "[cH:1]>>[c:1]Cl",                  # Aromatic H → Cl
        "[cH:1]>>[c:1]C",                   # Aromatic H → Me
        "[CH3:1]>>[CH2:1]F",                # Methyl → fluoromethyl
        "[NH2:1]>>[NH:1]C",                 # NH2 → NHMe
        "[OH:1]>>[O:1]C",                   # OH → OMe
        "[F:1]>>[Cl:1]",                    # F → Cl
        "[Cl:1]>>[F:1]",                    # Cl → F
        "[C:1](=O)[OH]>>[C:1](=O)N",       # COOH → CONH2
    ]

    candidates: list[tuple[str, str]] = []
    seen_smiles: set[str] = set()

    for parent_smi, parent_mol in ref_mols:
        seen_smiles.add(parent_smi)
        for rxn_smarts in REACTIONS:
            try:
                rxn = AllChem.ReactionFromSmarts(rxn_smarts)
                if rxn is None:
                    continue
                products = rxn.RunReactants((parent_mol,))
                for product_set in products[:5]:
                    for prod in product_set:
                        try:
                            Chem.SanitizeMol(prod)
                            new_smi = Chem.MolToSmiles(prod)
                            if new_smi and new_smi not in seen_smiles:
                                seen_smiles.add(new_smi)
                                candidates.append((new_smi, parent_smi))
                        except Exception:
                            continue
            except Exception:
                continue

        if len(candidates) >= n_candidates:
            break

    return candidates[:n_candidates]


# ---------------------------------------------------------------------------
# Strategy B: Fragment-Based Growing
# ---------------------------------------------------------------------------

def _fragment_grow(
    ref_mols: list[tuple[str, "Chem.Mol"]],
    n_candidates: int,
) -> list[tuple[str, str]]:
    """Generate novel molecules by recombining BRICS fragments from reference compounds.

    Returns list of (new_smiles, parent_smiles="fragment_recombination") tuples.
    """
    from rdkit import Chem
    from rdkit.Chem import BRICS

    # Decompose all reference molecules into fragments
    all_fragments: set[str] = set()
    for smi, mol in ref_mols:
        try:
            frags = BRICS.BRICSDecompose(mol, minFragmentSize=5)
            all_fragments.update(frags)
        except Exception:
            continue

    if len(all_fragments) < 2:
        return []

    # Recombine fragments
    frag_list = list(all_fragments)
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    try:
        # BRICS build generates an iterator of molecules from fragments
        builder = BRICS.BRICSBuild(
            [Chem.MolFromSmiles(f) for f in frag_list if Chem.MolFromSmiles(f) is not None]
        )
        for i, mol in enumerate(builder):
            if i >= n_candidates * 2:  # Generate extra, filter later
                break
            try:
                Chem.SanitizeMol(mol)
                smi = Chem.MolToSmiles(mol)
                if smi and smi not in seen:
                    seen.add(smi)
                    candidates.append((smi, "fragment_recombination"))
            except Exception:
                continue
    except Exception as e:
        logger.debug("BRICS build failed: %s", e)

    return candidates[:n_candidates]


# ---------------------------------------------------------------------------
# Strategy C: Matched Molecular Pair Transforms
# ---------------------------------------------------------------------------

def _mmp_transform(
    ref_mols: list[tuple[str, "Chem.Mol"]],
    n_candidates: int,
) -> list[tuple[str, str]]:
    """Apply known productive transforms to reference molecules.

    Uses a curated set of medicinal chemistry transforms that commonly
    improve CNS drug-like properties.

    Returns list of (new_smiles, parent_smiles) tuples.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Curated transforms for CNS optimization (from medicinal chemistry literature)
    TRANSFORMS = [
        # (SMARTS pattern, SMARTS product, description)
        ("[C:1]([F])([F])[F]", "[C:1]([F])([F])[H]", "CF3→CHF2: reduce lipophilicity"),
        ("[c:1]1[cH:2][cH:3][cH:4][cH:5][cH:6]1", "[c:1]1[cH:2][cH:3][nH:4][cH:5]1", "phenyl→pyridine: improve solubility"),
        ("[NH:1][CH3:2]", "[N:1]([CH3:2])[CH3]", "N-Me→N,N-diMe: block metabolism"),
        ("[OH:1]", "[OCH3:1]", "OH→OMe: improve metabolic stability"),
        ("[C:1](=O)[N:2]([H])[H]", "[C:1](=O)[N:2]([H])[CH3]", "amide→N-Me-amide: improve permeability"),
    ]

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    for parent_smi, parent_mol in ref_mols:
        for pat_smarts, prod_smarts, desc in TRANSFORMS:
            try:
                pattern = Chem.MolFromSmarts(pat_smarts)
                if pattern is None or not parent_mol.HasSubstructMatch(pattern):
                    continue

                replacement = Chem.MolFromSmiles(prod_smarts) or Chem.MolFromSmarts(prod_smarts)
                if replacement is None:
                    continue

                products = AllChem.ReplaceSubstructs(parent_mol, pattern, replacement)
                for prod in products[:2]:
                    try:
                        Chem.SanitizeMol(prod)
                        new_smi = Chem.MolToSmiles(prod)
                        if new_smi and new_smi not in seen and new_smi != parent_smi:
                            seen.add(new_smi)
                            candidates.append((new_smi, parent_smi))
                    except Exception:
                        continue
            except Exception:
                continue

    return candidates[:n_candidates]


# ---------------------------------------------------------------------------
# Scoring and filtering
# ---------------------------------------------------------------------------

def _score_and_filter(
    raw_candidates: list[tuple[str, str]],
    filters: dict,
    ref_mols: list[tuple[str, "Chem.Mol"]],
) -> list[GeneratedMolecule]:
    """Score candidates with molecular properties and filter by drug-likeness."""
    from rdkit import Chem
    from rdkit.Chem import DataStructs, AllChem

    from executors.molecular_executor import compute_drug_properties

    # Compute reference fingerprints for similarity
    ref_fps = []
    for smi, mol in ref_mols:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        ref_fps.append(fp)

    results: list[GeneratedMolecule] = []

    for new_smi, parent_smi in raw_candidates:
        props = compute_drug_properties(new_smi)
        if props is None:
            continue

        # Apply filters
        if props.cns_mpo < filters["cns_mpo_min"]:
            continue
        if props.lipinski_violations > filters["lipinski_max"]:
            continue
        if props.synthetic_accessibility > filters["sa_max"]:
            continue
        if props.molecular_weight < filters["mw_min"] or props.molecular_weight > filters["mw_max"]:
            continue

        # Compute similarity to parent
        mol = Chem.MolFromSmiles(new_smi)
        if mol:
            cand_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            similarities = DataStructs.BulkTanimotoSimilarity(cand_fp, ref_fps)
            max_sim = max(similarities) if similarities else 0.0
        else:
            max_sim = 0.0

        # Determine strategy from parent
        strategy = "fragment_grow" if parent_smi == "fragment_recombination" else "scaffold_hop"

        results.append(GeneratedMolecule(
            smiles=new_smi,
            parent_smiles=parent_smi,
            strategy=strategy,
            cns_mpo=props.cns_mpo,
            molecular_weight=props.molecular_weight,
            logp=props.logp,
            lipinski_violations=props.lipinski_violations,
            synthetic_accessibility=props.synthetic_accessibility,
            similarity_to_parent=round(max_sim, 3),
            is_novel=(max_sim < 0.85),  # Novel if <85% similar to any reference
        ))

    return results
