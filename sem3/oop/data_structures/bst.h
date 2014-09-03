#ifndef BST_H
#define BST_H

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
	_root = nullptr;
	_size = 0;
}

template<class T>
ds::Bst<T>::Bst(const Bst& bst)
{
	BstNode* tmpRoot = bst._root;

	_root = _copy(&tmpRoot);
	_size = bst.size();
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
	BstNode* parent = nullptr;

	while (node != nullptr)
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
	node->left = nullptr;
	node->right = nullptr;

	if (parent == nullptr)
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

	while (node != nullptr)
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
	BstNode* parent = nullptr;
	BstNode* tmpNode = nullptr;

	while (node != nullptr && node->key != key)
	{
		parent = node;

		if (node->key < key)
			node = node->right;
		else if (key < node->key)
			node = node->left;
	}

	if (node == nullptr)
		return;

	if (node->left == nullptr && node->right == nullptr)
	{
		if (parent == nullptr)
			_root = nullptr;
		else if (parent->key < key)
			parent->right = nullptr;
		else
			parent->left = nullptr;

		tmpNode = node;
	}
	else if (node->left != nullptr && node->right != nullptr)
	{
		tmpNode = node->left;
		parent = node;

		while (tmpNode->right != nullptr)
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
		if (node->left != nullptr)
			tmpNode = node->left;
		else
			tmpNode = node->right;

		if (parent == nullptr)
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
	BstNode* tmpRoot = bst._root;

	_root = _copy(&tmpRoot);
	_size = bst.size();

	return *this;
}

template<class T>
void ds::Bst<T>::_clear(BstNode** node)
{
	if (*node == nullptr)
		return;

	_clear(&(*node)->left);
	_clear(&(*node)->right);
	
	delete *node;

	*node = nullptr;
}

template<class T>
typename ds::Bst<T>::BstNode* ds::Bst<T>::_copy(BstNode** node)
{
	if (*node == nullptr)
		return nullptr;

	BstNode* tmpNode = new BstNode;
	tmpNode->key = (*node)->key;
	tmpNode->left = _copy(&(*node)->left);
	tmpNode->right = _copy(&(*node)->right);

	return tmpNode;
}

#endif
