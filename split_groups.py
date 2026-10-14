"""Connect sources that cannot be separated under declared sequence constraints."""
import hashlib
from collections import defaultdict

def canonical_group_ids(rows, groups):
    """Make equivalent partitions identical even when grouping labels are renamed."""
    if len(rows) != len(groups):
        raise ValueError('Row/group count mismatch')
    members=defaultdict(list)
    for row,group in zip(rows,groups): members[group].append(row['id'])
    signatures={g:tuple(sorted(ids)) for g,ids in members.items()}
    ordered=sorted(signatures,key=lambda g:signatures[g])
    mapping={g:i for i,g in enumerate(ordered)}
    return [mapping[g] for g in groups]

class UnionFind:
    def __init__(self, nodes):
        self.parent={n:n for n in nodes}
    def find(self, node):
        root=node
        while self.parent[root] != root:
            root=self.parent[root]
        while node != root:
            parent=self.parent[node]; self.parent[node]=root; node=parent
        return root
    def join(self, left, right):
        a,b=self.find(left),self.find(right)
        if a != b:
            self.parent[max(a,b)]=min(a,b)

def source_sequence_components(rows, sequence_edges=()):
    """Keep whole papers together, and merge papers sharing connected sequences.

    sequence_edges comes from an external similarity screen, with its own metric
    and threshold declared in the experiment. Components are a conservative
    grouping rule, not a novel graph partitioning algorithm.
    """
    papers=sorted({r['paper'] for r in rows})
    union=UnionFind(papers)
    owners=defaultdict(set)
    for row in rows:
        owners[row['sequence']].add(row['paper'])
    for sources in owners.values():
        sources=sorted(sources)
        for source in sources[1:]: union.join(sources[0],source)
    for a,b in sequence_edges:
        if a not in owners or b not in owners:
            raise ValueError('Similarity edge refers to an absent sequence')
        all_sources=sorted(owners[a] | owners[b])
        for source in all_sources[1:]: union.join(all_sources[0],source)
    groups=defaultdict(list)
    for paper in papers: groups[union.find(paper)].append(paper)
    ledger=[]
    assignment={}
    for component,paper_list in sorted(groups.items()):
        ident=hashlib.sha256('\n'.join(paper_list).encode()).hexdigest()[:12]
        ledger.append({'component':ident,'papers':paper_list,
                       'n_assay_rows':sum(r['paper'] in paper_list for r in rows),
                       'n_sequences':len({r['sequence'] for r in rows if r['paper'] in paper_list})})
        for paper in paper_list: assignment[paper]=ident
    return [assignment[r['paper']] for r in rows],ledger
