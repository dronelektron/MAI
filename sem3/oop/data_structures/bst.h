#ifndef BST_H
#define BST_H

#include <cstdlib>

namespace ds
{
	template<class T>
	class Bst
	{
	public:
		struct BstNode
		{
			T key;
			BstNode* left;
			BstNode* right;
		};

		class iterator
		{
		public:
			iterator();
			iterator(BstNode* node);

			iterator& operator++();
			bool operator!=(const iterator& it);
			T& operator*();
			T* operator->();

		private:
			BstNode* _root;
			BstNode* _cur;
		};

		Bst();
		Bst(const Bst& bst);
		~Bst();
		
		BstNode* insert(const T& key);
		BstNode* find(const T& key);
		void erase(const T& key);
		void clear();
		size_t size() const;
		bool empty() const;

		Bst& operator=(const Bst& bst);

		iterator begin();
		iterator end();

	private:
		BstNode* _root;
		size_t _size;

		void _clear(BstNode** node);
		BstNode* _copy(BstNode** node);
	};
}

template<class T>
ds::Bst<T>::Bst()
{
	_root = NULL;
	_size = 0;
}

template<class T>
ds::Bst<T>::Bst(const Bst& bst)
{
	_root = NULL;
	_size = 0;

	if (this != &bst)
	{
		BstNode* tmpRoot = bst._root;

		_root = _copy(&tmpRoot);
		_size = bst._size;
	}
}

template<class T>
ds::Bst<T>::~Bst()
{
	_clear(&_root);
}

template<class T>
typename ds::Bst<T>::BstNode* ds::Bst<T>::insert(const T& key)
{
	BstNode* node = _root;
	BstNode* parent = NULL;

	while (node != NULL)
	{
		parent = node;

		if (node->key < key)
			node = node->right;
		else if (key < node->key)
			node = node->left;
		else
			return node;
	}

	node = new BstNode;
	node->key = key;
	node->left = NULL;
	node->right = NULL;

	if (parent == NULL)
		_root = node;
	else if (parent->key < key)
		parent->right = node;
	else
		parent->left = node;

	_size++;

	return node;
}

template<class T>
typename ds::Bst<T>::BstNode* ds::Bst<T>::find(const T& key)
{
	BstNode* node = _root;

	while (node != NULL)
	{
		if (node->key < key)
			node = node->right;
		else if (key < node->key)
			node = node->left;
		else
			return node;
	}

	return node;
}

template<class T>
void ds::Bst<T>::erase(const T& key)
{
	BstNode* node = _root;
	BstNode* parent = NULL;
	BstNode* tmpNode = NULL;

	while (node != NULL && node->key != key)
	{
		parent = node;

		if (node->key < key)
			node = node->right;
		else if (key < node->key)
			node = node->left;
	}

	if (node == NULL)
		return;

	if (node->left == NULL && node->right == NULL)
	{
		if (parent == NULL)
			_root = NULL;
		else if (parent->key < key)
			parent->right = NULL;
		else
			parent->left = NULL;

		tmpNode = node;
	}
	else if (node->left != NULL && node->right != NULL)
	{
		tmpNode = node->left;
		parent = node;

		while (tmpNode->right != NULL)
		{
			parent = tmpNode;
			tmpNode = tmpNode->right;
		}

		node->key = tmpNode->key;

		if (parent == node)
			node->left = tmpNode->left;
		else
			parent->right = tmpNode->left;
	}
	else
	{
		if (node->left != NULL)
			tmpNode = node->left;
		else
			tmpNode = node->right;

		if (parent == NULL)
			_root = tmpNode;
		else if (node == parent->left)
			parent->left = tmpNode;
		else
			parent->right = tmpNode;

		tmpNode = node;
	}

	delete tmpNode;

	_size--;
}

template<class T>
void ds::Bst<T>::clear()
{
	_clear(&_root);

	_size = 0;
}

template<class T>
size_t ds::Bst<T>::size() const
{
	return _size;
}

template<class T>
bool ds::Bst<T>::empty() const
{
	return _size == 0;
}

template<class T>
ds::Bst<T>& ds::Bst<T>::operator=(const Bst& bst)
{
	if (this != &bst)
	{
		_clear(&_root);

		BstNode* tmpRoot = bst._root;

		_root = _copy(&tmpRoot);
		_size = bst.size();
	}

	return *this;
}

template<class T>
void ds::Bst<T>::_clear(BstNode** node)
{
	if (*node == NULL)
		return;

	_clear(&(*node)->left);
	_clear(&(*node)->right);

	delete *node;

	*node = NULL;
}

template<class T>
typename ds::Bst<T>::BstNode* ds::Bst<T>::_copy(BstNode** node)
{
	if (*node == NULL)
		return NULL;

	BstNode* tmpNode = new BstNode;
	tmpNode->key = (*node)->key;
	tmpNode->left = _copy(&(*node)->left);
	tmpNode->right = _copy(&(*node)->right);

	return tmpNode;
}

template<class T>
typename ds::Bst<T>::iterator ds::Bst<T>::begin()
{
	return iterator(_root);
}

template<class T>
typename ds::Bst<T>::iterator ds::Bst<T>::end()
{
	return iterator();
}

template<class T>
ds::Bst<T>::iterator::iterator()
{
	_cur = NULL;
}

template<class T>
ds::Bst<T>::iterator::iterator(BstNode* node)
{
	_root = node;

	if (node != NULL)
		while (node->left != NULL)
			node = node->left;

	_cur = node;
}

template<class T>
typename ds::Bst<T>::iterator& ds::Bst<T>::iterator::operator++()
{
	if (_cur->right != NULL)
	{
		_cur = _cur->right;

		while (_cur->left != NULL)
			_cur = _cur->left;
	}
	else
	{
		BstNode* succ = NULL;
		BstNode* root = _root;

		while (root != NULL)
		{
			if (_cur->key < root->key)
			{
				succ = root;
				root = root->left;
			}
			else if (_cur->key > root->key)
				root = root->right;
			else
				break;
		}

		_cur = succ;
	}

	return *this;
}

template<class T>
bool ds::Bst<T>::iterator::operator!=(const iterator& it)
{
	return _cur != it._cur;
}

template<class T>
T& ds::Bst<T>::iterator::operator*()
{
	return _cur->key;
}

template<class T>
T* ds::Bst<T>::iterator::operator->()
{
	return &_cur->key;
}

#endif
