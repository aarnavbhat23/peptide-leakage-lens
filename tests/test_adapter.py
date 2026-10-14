import unittest
from audit_public_data import prepare
from tool import analyze

class AdapterTests(unittest.TestCase):
    def test_collapse_only_within_same_paper(self):
        raw=b'sequence\tpaper_doi\nKLLK\tpaper-A\nKLLK\tpaper-A\nKLLK\tpaper-B\n'
        data,n=prepare(raw)
        self.assertEqual(n,3)
        self.assertEqual(len(data['rows']),2)
        self.assertEqual(data['rows'][0]['source_row_numbers'],[1,2])
        report=analyze(data)
        self.assertEqual(len(report['cross_split_pairs']),1)
        self.assertIsNone(report['cross_split_pairs'][0]['same_chemistry'])
    def test_missing_doi_fails(self):
        with self.assertRaises(ValueError):
            prepare(b'sequence\tpaper_doi\nKLLK\t\n')
