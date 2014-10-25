#ifndef SHAPE_H
#define SHAPE_H

#include <iostream>
#include "visitor.h"

class Shape
{
public:
	virtual ~Shape();

	virtual double accept(Visitor* vis) = 0;
	virtual void printInfo() = 0;
};

class Romb : public Shape
{
public:
	Romb();

	virtual double accept(Visitor* vis);
	virtual void printInfo();

	void setDiagHor(double val);
	void setDiagVer(double val);
	double getDiagHor() const;
	double getDiagVer() const;

private:
	double _diagHor;
	double _diagVer;
};

class Side5 : public Shape
{
public:
	Side5();

	virtual double accept(Visitor* vis);
	virtual void printInfo();

	void setSide(double side);
	double getSide() const;

protected:
	double _side;
};

class Side6 : public Side5
{
public:
	virtual double accept(Visitor* vis);
	virtual void printInfo();
};

#endif
