#ifndef VISITOR_H
#define VISITOR_H

#include <cmath>

class Romb;
class Side5;
class Side6;

class Visitor
{
public:
	virtual double visit(Romb* shape);
	virtual double visit(Side5* shape);
	virtual double visit(Side6* shape);
};

#endif
