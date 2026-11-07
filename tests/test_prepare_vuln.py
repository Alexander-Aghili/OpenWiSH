"""prepare_vuln: Juliet CWE-476 testcase -> a matched vulnerable and safe program, with no label leaks."""
import importlib.util
import re
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("prepare_vuln", Path(__file__).resolve().parents[1] / "scripts/prepare_vuln.py")
pv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv)

CASE = r'''/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE476_NULL_Pointer_Dereference__struct_21.c
*/
#include "std_testcase.h"
#include <wchar.h>
static const int STATIC_CONST_TRUE = 1; /* true */
#ifndef OMITBAD
static int badStatic = 0;
static void badSink(twoIntsStruct * data)
{
    if(badStatic)
    {
        printIntLine(data->intOne); // FLAW here
    }
}
void CWE476_NULL_Pointer_Dereference__struct_21_bad()
{
    twoIntsStruct * data;
    /* POTENTIAL FLAW: Set data to NULL */
    data = NULL;
    badStatic = 1; /* true */
    badSink(data);
}
#endif /* OMITBAD */
#ifndef OMITGOOD
static int goodB2G1Static = 0;
static int goodG2BStatic = 0;
static void goodB2G1Sink(twoIntsStruct * data)
{
    if(goodB2G1Static)
    {
        printLine("Benign, fixed string");
    }
    else
    {
        if (data != NULL)
        {
            printIntLine(data->intOne);
        }
    }
}
static void goodB2G1()
{
    twoIntsStruct * data;
    data = NULL;
    goodB2G1Static = 0; /* false */
    goodB2G1Sink(data);
}
static void goodG2BSink(twoIntsStruct * data)
{
    if(goodG2BStatic)
    {
        printIntLine(data->intOne);
    }
}
static void goodG2B()
{
    twoIntsStruct * data;
    twoIntsStruct tmpData;
    data = &tmpData;
    goodG2BStatic = 1;
    goodG2BSink(data);
}
void CWE476_NULL_Pointer_Dereference__struct_21_good()
{
    goodB2G1();
    goodG2B();
}
#endif /* OMITGOOD */
#ifdef INCLUDEMAIN
int main(int argc, char * argv[])
{
    CWE476_NULL_Pointer_Dereference__struct_21_good();
    CWE476_NULL_Pointer_Dereference__struct_21_bad();
    return 0;
}
#endif
'''

LEAK = re.compile(r"bad|good|G2B|B2G|FLAW|FIX|CWE|OMIT|INCLUDEMAIN", re.I)


class Build(unittest.TestCase):
    def test_pair_has_entry_and_no_label_leak(self):
        pair = pv.build_pair(CASE, seed=0)
        for kind in ("vulnerable", "safe"):
            src = pair[kind]
            self.assertIn("void entry()", src)
            self.assertIsNone(LEAK.search(src), f"{kind} leaks: {LEAK.search(src)}")
            self.assertNotIn("static void entry", src)
            self.assertNotIn("/*", src)
            self.assertNotIn("//", src)
            self.assertNotIn("int main", src)
            self.assertIn("STATIC_CONST_TRUE", src)  # shared preamble kept

    def test_safe_version_keeps_one_good_chain_only(self):
        pair = pv.build_pair(CASE, seed=0)
        safe = pair["safe"]
        self.assertIn("data = NULL;", safe)  # first good variant in the good() wrapper: goodB2G1, NULL but checked
        self.assertIn("if (data != NULL)", safe)
        self.assertNotIn("tmpData", safe)  # goodG2B chain dropped
        self.assertEqual(pair["good_variant"], "goodB2G1")

    def test_helpers_share_role_names_across_the_pair_and_change_with_the_seed(self):
        a, b = pv.build_pair(CASE, seed=0), pv.build_pair(CASE, seed=1)
        sink_a = re.search(r"static void (\w+)\(twoIntsStruct", a["vulnerable"]).group(1)
        self.assertIn(f"static void {sink_a}(twoIntsStruct", a["safe"])
        sink_b = re.search(r"static void (\w+)\(twoIntsStruct", b["vulnerable"]).group(1)
        self.assertNotEqual(sink_a, sink_b)

    def test_prompt_asks_the_question_about_entry(self):
        p = pv.question("void entry() {}")
        self.assertIn("entry()", p)
        self.assertTrue(p.rstrip().endswith("Answer YES or NO."))


if __name__ == "__main__":
    unittest.main()
