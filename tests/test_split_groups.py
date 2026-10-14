import unittest
from split_groups import source_sequence_components,canonical_group_ids

class ComponentTests(unittest.TestCase):
    def test_shared_sequence_merges_sources_transitively(self):
        rows=[{'paper':'A','sequence':'AAA'},{'paper':'B','sequence':'AAA'},
              {'paper':'B','sequence':'CCC'},{'paper':'C','sequence':'CCC'},
              {'paper':'D','sequence':'DDD'}]
        groups,ledger=source_sequence_components(rows)
        self.assertEqual(len(set(groups)),2)
        self.assertEqual(groups[0],groups[3])
        self.assertNotEqual(groups[0],groups[4])
    def test_declared_similarity_edge_connects_sources(self):
        rows=[{'paper':'A','sequence':'AAA'},{'paper':'B','sequence':'AAC'}]
        groups,_=source_sequence_components(rows,[('AAA','AAC')])
        self.assertEqual(groups[0],groups[1])
    def test_undeclared_edge_is_rejected(self):
        with self.assertRaises(ValueError):
            source_sequence_components([{'paper':'A','sequence':'AAA'}],[('AAA','XXX')])
    def test_group_labels_do_not_change_a_partition(self):
        rows=[{'id':str(i)} for i in range(5)]
        self.assertEqual(canonical_group_ids(rows,['A','A','B','C','C']),
                         canonical_group_ids(rows,['z','z','y','x','x']))
