//                 code-gen.cpp 2014
#include "code-gen.h"

using namespace std;

// S -> PROG
int tCG::p01()
{
	string header = "/*  " + lex.Authentication() +
	"   */\n#include \"mlisp.h\"\n";
	header += declarations + "//________________ \n";
	S1->obj = header + S1->obj;
	
	return 0;
}

// CALC -> E1
int tCG::p02()
{
	S1->obj = "display(" + S1->obj + "); newline();\n";

	return 0;
}

// CALC -> BOOL
int tCG::p03()
{
	S1->obj = "display(" + S1->obj + "); newline();\n";

	return 0;
}

// CALC -> STR
int tCG::p04()
{
	S1->obj = "display(" + S1->obj + "); newline();\n";

	return 0;
}

// CALC -> DISP
int tCG::p05()
{
	S1->obj += "; newline();\n";

	return 0;
}

// DISP -> ( display E1 )
int tCG::p06()
{
	S1->obj = "display(" + S3->obj + ")";

	return 0;
}

// DISP -> ( display BOOL )
int tCG::p07()
{
	S1->obj = "display(" + S3->obj + ")";

	return 0;
}

// DISP -> ( display STR )
int tCG::p08()
{
	S1->obj = "display(" + S3->obj + ")";

	return 0;
}

// DISP -> ( newline )
int tCG::p09()
{
	S1->obj = "newline()";

	return 0;
}

// PRED -> HPRED BOOL )
int tCG::p10()
{
	S1->obj += S2->obj + ";\n}}\n";

	return 0;
}

// HPRED -> PDPAR )
int tCG::p11()
{
	S1->obj += ")";
	declarations += S1->obj + ";\n";
	S1->obj += "\n{{\nreturn ";

	return 0;
}

// PDPAR -> ( define ( $idq
int tCG::p12()
{
	S1->obj = "bool " + decor(S4->name) + "(";
	S1->count = 0;

	return 0;
}

// PDPAR -> PDPAR $id
int tCG::p13()
{
	if (S1->count > 0)
		S1->obj += ", ";

	S1->obj += "double " + decor(S2->name);
	++S1->count;

	return 0;
}

// CPROC -> HCPROC )
int tCG::p14()
{
	S1->obj += ")";
	
	return 0;
}

// HCPROC -> ( $id
int tCG::p15()
{
	S1->obj = decor(S2->name) + "(";
	S1->count = 0;

	return 0;
}

// HCPROC -> HCPROC E
int tCG::p16()
{
	if (S1->count > 0)
		S1->obj += ", ";

	S1->obj += S2->obj;
	++S1->count;

	return 0;
}

// CPRED -> HCPRED )
int tCG::p17()
{
	S1->obj += ")";
	
	return 0;
}

// HCPRED -> ( $idq
int tCG::p18()
{
	S1->obj = decor(S2->name) + "(";
	S1->count = 0;

	return 0;
}

// HCPRED -> HCPRED E
int tCG::p19()
{
	if (S1->count > 0)
		S1->obj += ", ";

	S1->obj += S2->obj;
	++S1->count;

	return 0;
}

// E -> $float
int tCG::p20()
{
	S1->obj = S1->name;

	return 0;
}

// E -> DINT
int tCG::p21()
{
	return 0;
}

// E -> $id
int tCG::p22()
{
	S1->obj = decor(S1->name);

	return 0;
}

// E -> ADD
int tCG::p23()
{
	return 0;
}

// E -> SUB
int tCG::p24()
{
	return 0;
}

// E -> DIV
int tCG::p25()
{
	return 0;
}

// E -> MUL
int tCG::p26()
{
	return 0;
}

// E -> COND
int tCG::p27()
{
	return 0;
}

// E -> CPROC
int tCG::p28()
{
	return 0;
}

// DINT -> $int
int tCG::p29()
{
	S1->obj = S1->name;

	return 0;
}

// DINT -> $oct
int tCG::p30()
{
	int i = 0;

	S1->obj = "";

	if (S1->name[0] == '-')
	{
		for (i = 1; i < S1->name.length() - 1 && S1->name[i] == '0'; ++i);

		S1->obj = "-";
	}
	else
		for (; i < S1->name.length() - 1 && S1->name[i] == '0'; ++i);

	S1->obj += &S1->name.c_str()[i];

	return 0;
}

// ADD -> HADD E1 )
int tCG::p31()
{
	S1->obj += S2->obj + ")";
	
	return 0;
}

// HADD -> ( +
int tCG::p32()
{
	S1->obj = "(";

	return 0;
}

// HADD -> HADD E1
int tCG::p33()
{
	S1->obj += S2->obj + " + ";

	return 0;
}

// MUL -> HMUL E1 )
int tCG::p34()
{
	S1->obj += S2->obj + ")";
	
	return 0;
}

// HMUL -> ( *
int tCG::p35()
{
	S1->obj = "(";

	return 0;
}

// HMUL -> HMUL E1
int tCG::p36()
{
	S1->obj += S2->obj + " * ";

	return 0;
}

// SUB -> HSUB E1 )
int tCG::p37()
{
	if (S1->count == 1)
		S1->obj += "-" + S2->obj + ")";
	else
		S1->obj += S2->obj + ")";
	
	return 0;
}

// HSUB -> ( -
int tCG::p38()
{
	S1->obj = "(";
	S1->count = 1;

	return 0;
}

// HSUB -> HSUB E1
int tCG::p39()
{
	S1->obj += S2->obj + " - ";
	++S1->count;

	return 0;
}

// DIV -> HDIV E1 )
int tCG::p40()
{
	if (S1->count == 1)
		S1->obj += "1 / " + S2->obj + ")";
	else
		S1->obj += S2->obj + ")";
	
	return 0;
}

// HDIV -> ( /
int tCG::p41()
{
	S1->obj = "((double)";
	S1->count = 1;

	return 0;
}

// HDIV -> HDIV E1
int tCG::p42()
{
	S1->obj += S2->obj + " / ";
	++S1->count;
	
	return 0;
}

// BOOL -> $bool
int tCG::p43()
{
	if (S1->name == "#t")
		S1->obj = "true";
	else
		S1->obj = "false";
	
	return 0;
}

// BOOL -> CPRED
int tCG::p44()
{
	return 0;
}

// BOOL -> REL
int tCG::p45()
{
	return 0;
}

// BOOL -> OR
int tCG::p46()
{
	return 0;
}

// BOOL -> ( not BOOL )
int tCG::p47()
{
	S1->obj = "(!" + S3->obj + ")";

	return 0;
}

// REL -> HREL E1 )
int tCG::p48()
{
	S1->obj += S2->obj + ")";
	
	return 0;
}

// HREL -> ( > E
int tCG::p49()
{
	S1->obj = "(" + S3->obj + " > ";

	return 0;
}

// OR -> HOR BOOL )
int tCG::p50()
{
	S1->obj += S2->obj + ")";
	
	return 0;
}

// HOR -> ( or
int tCG::p51()
{
	S1->obj = "(";

	return 0;
}

// HOR -> HOR BOOL
int tCG::p52()
{
	S1->obj += S2->obj + " || ";
	
	return 0;
}

// COND -> HCOND ELSE )
int tCG::p53()
{
	S1->obj += S2->obj;

	return 0;
}

// HCOND -> ( cond
int tCG::p54()
{
	return 0;
}

// HCOND -> HCOND CLAUS
int tCG::p55()
{
	S1->obj += S2->obj;

	return 0;
}

// CLAUS -> HCLAUS E1 )
int tCG::p56()
{
	S1->obj += "(" + S2->obj + ") :\n";

	return 0;
}

// HCLAUS -> ( BOOL
int tCG::p57()
{
	S1->obj = "(" + S2->obj + ") ? ";

	return 0;
}

// ELSE -> HELSE E1 )
int tCG::p58()
{
	if (S1->count > 0)
		S1->obj += ",\n";

	S1->obj += S2->obj + ")";

	return 0;
}

// HELSE -> ( else
int tCG::p59()
{
	S1->obj = "(";
	S1->count = 0;

	return 0;
}

// HELSE -> HELSE DISP
int tCG::p60()
{
	if (S1->count > 0)
		S1->obj += ",\n";

	S1->obj += S2->obj;
	++S1->count;

	return 0;
}

// E1 -> E
int tCG::p61()
{
	return 0;
}

// STR -> $str
int tCG::p62()
{
	S1->obj = S1->name;

	return 0;
}

// SET -> HSET E1 )
int tCG::p63()
{
	S1->obj += S2->obj + ";\n";

	return 0;
}

// HSET -> ( set! $id
int tCG::p64()
{
	S1->obj = decor(S3->name) + " = ";

	return 0;
}

// VAR -> HVAR E1 )
int tCG::p65()
{
	S1->obj += " = " + S2->obj + ";\n";

	return 0;
}

// HVAR -> ( define $id
int tCG::p66()
{
	S1->obj = "double " + decor(S3->name);

	return 0;
}

// PROC -> HPROC E1 )
int tCG::p67()
{
	S1->obj += "return " + S2->obj + ";\n}}\n";

	return 0;
}

// HPROC -> PCPAR )
int tCG::p68()
{
	S1->obj += ")";
	declarations += S1->obj + ";\n";
	S1->obj += "\n{{\n";

	return 0;
}

// HPROC -> HPROC VAR
int tCG::p69()
{
	S1->obj += S2->obj;

	return 0;
}

// HPROC -> HPROC SET
int tCG::p70()
{
	S1->obj += S2->obj;
	
	return 0;
}

// HPROC -> HPROC DISP
int tCG::p71()
{
	S1->obj += S2->obj + ";\n";

	return 0;
}

// PCPAR -> ( define ( $id
int tCG::p72()
{
	S1->obj = "double " + decor(S4->name) + "(";
	S1->count = 0;
	
	return 0;
}

// PCPAR -> PCPAR $id
int tCG::p73()
{
	if (S1->count > 0)
		S1->obj += ", ";

	S1->obj += "double " + decor(S2->name);
	++S1->count;

	return 0;
}

// DEF -> PRED
int tCG::p74()
{
	return 0;
}

// DEF -> VAR
int tCG::p75()
{
	int pos = static_cast<int>(S1->obj.find("=")) - 1;

	while (pos >= 0 && S1->obj[pos] == ' ')
		--pos;

	declarations += "extern " + S1->obj.substr(0, pos + 1) + ";\n";

	return 0;
}

// DEF -> PROC
int tCG::p76()
{
	return 0;
}

// DEFS -> DEF
int tCG::p77()
{
	return 0;
}

// DEFS -> DEFS DEF
int tCG::p78()
{
	S1->obj += S2->obj;

	return 0;
}

// CALCS -> CALC
int tCG::p79()
{
	return 0;
}

// CALCS -> CALCS CALC
int tCG::p80()
{
	S1->obj += S2->obj;

	return 0;
}

// CALCS1 -> CALCS
int tCG::p81()
{
	return 0;
}

// PROG -> CALCS1
int tCG::p82()
{
	S1->obj = "int main()\n{\n" + S1->obj + "return 0;\n}\n";

	return 0;
}

// PROG -> DEFS
int tCG::p83()
{
	S1->obj += "int main()\n{\ndisplay(\"No calculations!\\n\");\nreturn 0;\n}\n";

	return 0;
}

// PROG -> DEFS CALCS1
int tCG::p84()
{
	S1->obj += "int main()\n{\n" + S2->obj + "return 0;\n}\n";

	return 0;
}

int tCG::p85() {return 0;}
int tCG::p86() {return 0;}
int tCG::p87() {return 0;}
int tCG::p88() {return 0;}
int tCG::p89() {return 0;}
int tCG::p90() {return 0;}
int tCG::p91() {return 0;}
int tCG::p92() {return 0;}
int tCG::p93() {return 0;}
int tCG::p94() {return 0;}
int tCG::p95() {return 0;}
int tCG::p96() {return 0;}
int tCG::p97() {return 0;}
int tCG::p98() {return 0;}
int tCG::p99() {return 0;}
int tCG::p100() {return 0;}
