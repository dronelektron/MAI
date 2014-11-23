#ifndef SHAPE_H
#define SHAPE_H

#include <iostream>
#include <random>
#include "visitor.h"

class Shape
{
public:
	virtual double accept(Visitor* vis) = 0;
	virtual void printInfo() = 0;
	virtual void randomize(std::default_random_engine& rnd, std::uniform_real_distribution<double>& urd) = 0;
};

class Romb : public Shape
{
public:
	virtual double accept(Visitor* vis);
	virtual void printInfo();
	virtual void randomize(std::default_random_engine& rnd, std::uniform_real_distribution<double>& urd);

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
	virtual void randomize(std::default_random_engine& rnd, std::uniform_real_distribution<double>& urd);

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
